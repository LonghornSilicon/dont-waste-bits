"""
TinyLlama-1.1B (GQA architecture) cross-architecture beta* calibration.
Session 26: 4th model family — tests Grouped Query Attention (GQA) where
n_kv_heads=4, head_dim=64 → k/v_proj output shape is (seq, 256) vs
q_proj (seq, 2048). Same hook structure as SmolLM (k_proj/v_proj) but
GQA reduces K/V dimensionality 8x.

Hypothesis: beta* = gap_mean/0.267 holds even for GQA architectures.
Prediction: gap_mean will be between SmolLM-360M (0.337) and SmolLM-1.7B (0.424)
since TinyLlama-1.1B is in-between in scale.
"""
import json
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CACHE_PATH = Path(__file__).parent.parent / "results" / "tinyllama_cal_cache.pt"
RESULT_PATH = Path(__file__).parent.parent / "results" / "tinyllama_cal_results.json"
BIT_CLASSES = [4, 8]
FPGA_COSTS = torch.tensor([0.29, 0.56], dtype=torch.float32)
N_TEXTS = 10
MAX_LEN = 64
EPOCHS, LR, BATCH_SZ = 10, 1e-3, 256
# Focus sweep around predicted range (between 360M=1.261 and 1.7B=1.584)
BETAS = [0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
N_SEEDS = 3
MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def quant_quality(x: torch.Tensor, bits: int) -> torch.Tensor:
    n_levels = 2 ** bits - 1
    half = n_levels // 2
    scale = x.abs().max().clamp(min=1e-8) / half
    x_q = (x / scale).round().clamp(-half, half) * scale
    norm = x.norm().clamp(min=1e-8)
    return 1.0 - (x_q - x).norm() / norm


def extract_kv_signals():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    print(f"Loading TinyLlama-1.1B (BF16) from {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers       # 22
    n_heads = cfg.num_attention_heads      # 32
    n_kv_heads = cfg.num_key_value_heads   # 4 (GQA)
    hidden_dim = cfg.hidden_size           # 2048
    head_dim = hidden_dim // n_heads       # 64
    kv_dim = n_kv_heads * head_dim         # 256

    print(f"TinyLlama: {n_layers} layers, hidden={hidden_dim}, "
          f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"  GQA: k_proj/v_proj output dim per token: {kv_dim}")

    all_q4, all_q8 = [], []
    layer_kv = {i: {"k": [], "v": []} for i in range(n_layers)}

    def make_k_hook(i):
        def hook(module, inp, out):
            # GQA: out shape is (batch, seq, n_kv_heads * head_dim) = (1, seq, 256)
            layer_kv[i]["k"].append(out.detach().float())
        return hook

    def make_v_hook(i):
        def hook(module, inp, out):
            layer_kv[i]["v"].append(out.detach().float())
        return hook

    hooks = []
    for i in range(n_layers):
        hooks.append(model.model.layers[i].self_attn.k_proj.register_forward_hook(make_k_hook(i)))
        hooks.append(model.model.layers[i].self_attn.v_proj.register_forward_hook(make_v_hook(i)))

    print(f"Loading WikiText-2 ({N_TEXTS} texts, max_len={MAX_LEN})...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                      trust_remote_code=True)
    texts = [r["text"] for r in ds if len(r["text"].split()) > 20][:N_TEXTS]

    t0 = time.time()
    with torch.no_grad():
        for ti, text in enumerate(texts):
            enc = tok(text, return_tensors="pt", max_length=MAX_LEN, truncation=True)
            for i in range(n_layers):
                layer_kv[i]["k"].clear()
                layer_kv[i]["v"].clear()
            model(**enc)
            n_tok = 0
            for i in range(n_layers):
                if not layer_kv[i]["k"]:
                    continue
                k = layer_kv[i]["k"][0][0]  # [seq_len, kv_dim]
                v = layer_kv[i]["v"][0][0]
                for t in range(k.shape[0]):
                    q4 = quant_quality(k[t], 4)
                    q8 = quant_quality(k[t], 8)
                    all_q4.append(float(q4))
                    all_q8.append(float(q8))
                    q4v = quant_quality(v[t], 4)
                    q8v = quant_quality(v[t], 8)
                    all_q4.append(float(q4v))
                    all_q8.append(float(q8v))
                    n_tok += 1
            print(f"  Text {ti+1}/{N_TEXTS}: {enc['input_ids'].shape[1]} tokens, "
                  f"{n_tok} kv-pairs/layer processed in {time.time()-t0:.1f}s")

    for h in hooks:
        h.remove()
    del model

    q4t = torch.tensor(all_q4, dtype=torch.float32)
    q8t = torch.tensor(all_q8, dtype=torch.float32)
    gap = (q8t - q4t).numpy()
    gap_mean = float(gap.mean())
    gap_std = float(gap.std())
    n_tokens = len(all_q4)

    print(f"\nExtracted {n_tokens} token signals in {time.time()-t0:.1f}s")
    print(f"gap_mean={gap_mean:.4f}  gap_std={gap_std:.4f}")
    print(f"Predicted beta* = {gap_mean:.4f}/0.267 = {gap_mean/0.267:.3f}")

    q_local = torch.stack([q4t, q8t], dim=1)
    signals = q_local.clone()

    torch.save({"signals": signals, "q_local": q_local,
                "gap_mean": gap_mean, "gap_std": gap_std,
                "n_tokens": n_tokens, "model": MODEL_ID},
               CACHE_PATH)
    print(f"Cached to {CACHE_PATH}")
    return signals, q_local, gap_mean


class BinaryFPGAController(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
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
            return [BIT_CLASSES[i] for i in self.net(x).argmax(dim=-1).tolist()]


def train_and_eval(signals, q_local, beta, seed=0):
    torch.manual_seed(seed)
    ctrl = BinaryFPGAController()
    opt = torch.optim.Adam(ctrl.parameters(), lr=LR)
    N = signals.shape[0]
    tau_sched = torch.linspace(2.0, 0.1, EPOCHS)
    for ep in range(EPOCHS):
        tau = tau_sched[ep].item()
        idx = torch.randperm(N)
        for start in range(0, N, BATCH_SZ):
            bs = signals[idx[start:start+BATCH_SZ]]
            bq = q_local[idx[start:start+BATCH_SZ]]
            opt.zero_grad()
            probs = ctrl(bs, tau=tau)
            loss = ((1.0 - (probs * bq).sum(-1)).mean()
                    + beta * (probs * FPGA_COSTS).sum(-1).mean() / 1.01)
            loss.backward()
            opt.step()
    bits = ctrl.predict_bits(signals)
    c = Counter(bits)
    p4 = round(100.0 * c.get(4, 0) / len(bits), 1)
    fpga = p4/100 * 0.290 + (1-p4/100) * 0.560
    return p4, round(1.010/fpga, 3)


def run_sweep(signals, q_local, gap_mean):
    predicted_beta_star = gap_mean / 0.267
    print(f"\n{'='*60}")
    print(f"Sweeping {len(BETAS)} betas x {N_SEEDS} seeds")
    print(f"Predicted beta* = {predicted_beta_star:.3f}")
    print(f"{'='*60}\n")
    t0 = time.time()
    results = {}
    for beta in BETAS:
        p4s = [train_and_eval(signals, q_local, beta, seed=s)[0]
               for s in range(N_SEEDS)]
        mean_p4 = float(np.mean(p4s))
        fpga = mean_p4/100 * 0.290 + (1-mean_p4/100) * 0.560
        speedup = round(1.010/fpga, 3)
        threshold = round(beta * 0.267, 4)
        regime = "8-bit" if mean_p4 < 5 else ("4-bit" if mean_p4 > 95 else "MIXED")
        print(f"  beta={beta:.2f}  threshold={threshold:.4f}  gap={gap_mean:.4f}  "
              f"p4={mean_p4:.1f}%  speedup={speedup:.2f}x  [{regime}]")
        results[str(beta)] = {
            "beta": beta, "threshold": threshold,
            "p4_runs": p4s, "p4_mean": round(mean_p4, 1),
            "speedup": speedup, "regime": regime,
        }

    sorted_betas = sorted(results.keys(), key=float)
    transition_lo = transition_hi = None
    for i, bk in enumerate(sorted_betas[:-1]):
        r0 = results[bk]["p4_mean"]
        r1 = results[sorted_betas[i+1]]["p4_mean"]
        if r0 < 5 and r1 >= 5:
            transition_lo = float(bk)
            transition_hi = float(sorted_betas[i+1])

    print(f"\n{'='*60}")
    print(f"RESULTS — TinyLlama-1.1B GQA")
    print(f"  gap_mean = {gap_mean:.4f}")
    print(f"  Predicted beta* = {predicted_beta_star:.3f}")
    error = None
    if transition_lo is not None:
        mid = (transition_lo + transition_hi) / 2
        error = abs(predicted_beta_star - mid)
        confirmed = error <= 0.04
        print(f"  Measured transition: [{transition_lo}, {transition_hi}]")
        print(f"  Midpoint: {mid:.3f}")
        print(f"  Error vs theory: {error:.3f}  "
              f"({'CONFIRMED <=0.04' if confirmed else f'outside +-0.04'})")
    else:
        print("  No transition found in sweep range")
    print(f"{'='*60}")

    output = {
        "model": MODEL_ID,
        "architecture": "TinyLlama-1.1B (GQA: n_kv_heads=4, n_heads=32, head_dim=64)",
        "n_kv_heads": 4,
        "n_heads": 32,
        "kv_dim_per_token": 256,
        "gap_mean": round(gap_mean, 4),
        "predicted_beta_star": round(predicted_beta_star, 3),
        "measured_transition": ([transition_lo, transition_hi]
                                if transition_lo is not None else None),
        "theory_error": round(error, 3) if error is not None else None,
        "formula_confirmed_0_04": (error <= 0.04) if error is not None else False,
        "elapsed_s": round(time.time()-t0, 1),
        "sweep": results,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {RESULT_PATH}")
    return output


def main():
    if CACHE_PATH.exists():
        print(f"Loading cached signals from {CACHE_PATH}")
        d = torch.load(CACHE_PATH, weights_only=True)
        signals, q_local = d["signals"], d["q_local"]
        gap_mean = float(d["gap_mean"])
        print(f"TinyLlama: {signals.shape[0]} tokens  gap_mean={gap_mean:.4f}  "
              f"predicted beta*={gap_mean/0.267:.3f}")
    else:
        signals, q_local, gap_mean = extract_kv_signals()

    run_sweep(signals, q_local, gap_mean)


if __name__ == "__main__":
    main()
