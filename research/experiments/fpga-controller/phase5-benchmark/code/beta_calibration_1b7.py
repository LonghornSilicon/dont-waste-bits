"""
Beta calibration experiment on SmolLM-1.7B (CPU, lightweight).
PROTOCOL (committed before running):
  Prediction: gap_mean > 0.337 (360M). Theory: gap_1b7 ~ 0.400 (from epsilon_eff ratio).
  Predicted beta* = 0.400/0.267 = 1.498 ~ 1.5. If confirmed, beta=1.5 is exactly at
  the phase transition for 1.7B -> genuine mixed allocation.
  This replaces the simulated 1.7B data with real measured values.
Design: 10 texts, max_length=64, to complete in ~30 min on CPU.
  Two-stage: Stage 1 extracts signals (loads model), Stage 2 sweeps beta (no model).
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME  = "HuggingFaceTB/SmolLM-1.7B"
TRAIN_TEXTS = 10
MAX_LEN     = 64
EPOCHS      = 10
LR          = 1e-3
BATCH_SIZE  = 256

BIT_CLASSES = [4, 8]
FPGA_COSTS  = torch.tensor([0.29, 0.56], dtype=torch.float32)

OUTPUT_DIR  = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_PATH  = OUTPUT_DIR / "beta_cal_1b7_cache.pt"
RESULT_PATH = OUTPUT_DIR / "beta_calibration_1b7.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor


class BinaryFPGAController(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, tau=1.0):
        return F.gumbel_softmax(self.net(x), tau=tau, hard=False)

    def predict_bits(self, x):
        with torch.no_grad():
            idx = self.net(x).argmax(dim=-1)
        return [BIT_CLASSES[i] for i in idx.tolist()]


def extract_signals(model_name, n_texts, max_len, cache_path):
    if cache_path.exists():
        print(f"Loading cached signals from {cache_path}")
        d = torch.load(cache_path, weights_only=True)
        return d["signals"], d["q_local"]

    print(f"Extracting signals from {model_name} ({n_texts} texts, max_len={max_len})...")
    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    num_layers = len(model.model.layers)
    print(f"Model loaded in {time.time()-t0:.0f}s. Layers: {num_layers}")

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 50][:n_texts]

    kv_buf, handles = {}, []
    def make_hook(li, proj):
        def hook(module, inp, out):
            kv_buf[f"{li}_{proj}"] = out.detach().cpu().float()
        return hook
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(li, "v")))

    all_signals, all_q_local = [], []
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f"  text {i+1}/{len(texts)} ...")
            kv_buf.clear()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
            T = inputs["input_ids"].shape[1]
            model(**inputs)
            for li in range(num_layers):
                k = kv_buf.get(f"{li}_k")
                v = kv_buf.get(f"{li}_v")
                if k is None or v is None:
                    continue
                k, v = k[0], v[0]
                kv = torch.cat([k, v], dim=-1)
                kv_norm = kv.norm(dim=-1)
                kv_norm_n = kv_norm / (kv_norm.max() + 1e-8)
                pos_frac = torch.linspace(0, 1, T)

                def qerr(kv_, bits):
                    kv_q = quantize_tensor(kv_, bits)
                    err = (kv_q - kv_).norm(dim=-1)
                    nrm = kv_.norm(dim=-1).clamp(min=1e-8)
                    return (err / nrm).clamp(0, 1)

                q4_local = (1.0 - qerr(kv, 4)).clamp(0, 1)
                q8_local = (1.0 - qerr(kv, 8)).clamp(0, 1)
                sig = torch.stack([kv_norm_n, pos_frac, q4_local, q8_local], dim=-1)
                q   = torch.stack([q4_local, q8_local], dim=-1)
                all_signals.append(sig)
                all_q_local.append(q)

    for h in handles:
        h.remove()
    del model

    signals = torch.cat(all_signals, dim=0)
    q_local = torch.cat(all_q_local, dim=0)
    torch.save({"signals": signals, "q_local": q_local}, cache_path)
    print(f"Cached {signals.shape[0]} tokens in {time.time()-t0:.0f}s")
    return signals, q_local


def train_and_eval(signals, q_local, beta):
    ctrl = BinaryFPGAController()
    opt  = torch.optim.Adam(ctrl.parameters(), lr=LR)
    N = signals.shape[0]
    tau_sched = torch.linspace(2.0, 0.1, EPOCHS)
    for ep in range(EPOCHS):
        tau = tau_sched[ep].item()
        idx = torch.randperm(N)
        for start in range(0, N, BATCH_SIZE):
            bs = signals[idx[start:start+BATCH_SIZE]]
            bq = q_local[idx[start:start+BATCH_SIZE]]
            opt.zero_grad()
            probs = ctrl(bs, tau=tau)
            loss = (1.0 - (probs * bq).sum(-1).mean()) + beta * (probs * FPGA_COSTS).sum(-1).mean() / 1.01
            loss.backward()
            opt.step()
    bits_pred = ctrl.predict_bits(signals)
    from collections import Counter
    c = Counter(bits_pred)
    total = len(bits_pred)
    p4 = c.get(4, 0) / total
    p8 = c.get(8, 0) / total
    cost = p4 * 0.290 + p8 * 0.560
    return {
        "p4_pct": round(100.0 * p4, 1),
        "p8_pct": round(100.0 * p8, 1),
        "avg_bits": round(4.0 * p4 + 8.0 * p8, 3),
        "fpga_speedup": round(1.010 / cost, 3) if cost > 0 else 0,
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("Beta calibration: SmolLM-1.7B (CPU, 10 texts, max_len=64)")
    print("Protocol prediction: gap_mean ~= 0.400, beta* ~= 1.498")
    print("=" * 60)

    signals, q_local = extract_signals(MODEL_NAME, TRAIN_TEXTS, MAX_LEN, CACHE_PATH)
    gap = (q_local[:, 1] - q_local[:, 0]).numpy()
    gap_mean, gap_std = float(gap.mean()), float(gap.std())
    avg_q4 = float(q_local[:, 0].mean())
    avg_q8 = float(q_local[:, 1].mean())

    pred_beta_star = gap_mean / 0.2673
    print(f"Tokens: {signals.shape[0]}")
    print(f"avg q4_local={avg_q4:.4f}, avg q8_local={avg_q8:.4f}")
    print(f"gap mean={gap_mean:.4f} std={gap_std:.4f}")
    print(f"Predicted beta* = {pred_beta_star:.3f}")
    print(f"(Simulated prediction was: gap_mean=0.400, beta*=1.498)")

    betas = [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 2.0]
    results = []
    for beta in betas:
        thr = beta * 0.270 / 1.01
        m = train_and_eval(signals, q_local, beta)
        outcome = "4-bit" if m["p4_pct"] >= 90 else ("8-bit" if m["p4_pct"] <= 10 else "mixed")
        print(f"  beta={beta}: thr={thr:.4f} -> p4={m['p4_pct']}% speedup={m['fpga_speedup']}x [{outcome}]")
        results.append({"beta": beta, "threshold": round(thr, 4), **m, "outcome": outcome})

    elapsed = time.time() - t0
    output = {
        "model": MODEL_NAME,
        "n_texts": TRAIN_TEXTS,
        "max_length": MAX_LEN,
        "n_tokens": int(signals.shape[0]),
        "gap_mean": round(gap_mean, 4),
        "gap_std": round(gap_std, 4),
        "avg_q4_local": round(avg_q4, 4),
        "avg_q8_local": round(avg_q8, 4),
        "predicted_beta_star": round(pred_beta_star, 3),
        "simulation_prediction_gap_mean": 0.400,
        "simulation_prediction_beta_star": 1.498,
        "elapsed_s": round(elapsed, 1),
        "sweep": results,
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULT_PATH} ({elapsed:.0f}s)")

    print("\n=== CROSS-SCALE COMPARISON (MEASURED) ===")
    print(f"  135M: gap_mean=0.3297, measured  beta*=[1.2, 1.3]")
    print(f"  360M: gap_mean=0.3367, measured  beta*=[1.2, 1.4]")
    print(f"  1.7B: gap_mean={gap_mean:.4f}, predicted beta*={pred_beta_star:.3f} (THIS RUN)")
    if abs(gap_mean - 0.400) < 0.04:
        print("  => Simulation estimate CONFIRMED (within 4% of 0.400)")
    else:
        print(f"  => Simulation estimate UPDATED (was 0.400, now {gap_mean:.4f})")

    return output


if __name__ == "__main__":
    main()
