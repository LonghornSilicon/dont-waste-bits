"""
Smoke test: run per-token pertok pipeline on SmolLM-360M (CPU, small scale).
Validates code path end-to-end. Expected result: ~100% 4-bit (eff_residual=8.1% < threshold).
This is the CORRECT result for 360M — confirms the code works before 1.7B GPU run.
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME  = "HuggingFaceTB/SmolLM-360M"
TRAIN_TEXTS = 30       # small for CPU smoke test
EPOCHS      = 10
LR          = 1e-3
BATCH_SIZE  = 256
EVAL_TEXTS  = 50       # HellaSwag-style quick check via wikitext perplexity proxy

BIT_CLASSES = [4, 8]
FPGA_COSTS  = torch.tensor([0.29, 0.56], dtype=torch.float32)

OUTPUT_DIR  = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_PATH  = OUTPUT_DIR / "smoke_360m_pertok_cache.pt"
RESULT_PATH = OUTPUT_DIR / "smoke_360m_pertok_results.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor


class BinaryFPGAControllerPertok(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, tau=1.0, hard=False):
        logits = self.net(x)
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard)
        return probs, logits

    def predict_bits(self, x):
        with torch.no_grad():
            indices = self.net(x).argmax(dim=-1)
        return [BIT_CLASSES[i] for i in indices.tolist()]


def quantize_rel_error(kv: torch.Tensor, bits: int) -> torch.Tensor:
    kv_q = quantize_tensor(kv, bits)
    err = (kv_q - kv).norm(dim=-1)
    nrm = kv.norm(dim=-1).clamp(min=1e-8)
    return (err / nrm).clamp(0, 1)


def extract_signals(model_name, n_texts, cache_path, device="cpu"):
    if cache_path.exists():
        print(f"Loading cached signals from {cache_path}")
        data = torch.load(cache_path, weights_only=True)
        return data["signals"], data["q_local"]

    print(f"Extracting signals from {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    num_layers = len(model.model.layers)

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 50][:n_texts]

    kv_buf = {}
    handles = []
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
            if i % 10 == 0:
                print(f"  text {i}/{len(texts)}")
            kv_buf.clear()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
            T = inputs["input_ids"].shape[1]
            model(**inputs)
            for li in range(num_layers):
                k = kv_buf.get(f"{li}_k")
                v = kv_buf.get(f"{li}_v")
                if k is None or v is None:
                    continue
                k, v = k[0], v[0]
                kv = torch.cat([k, v], dim=-1)
                kv_norm   = kv.norm(dim=-1)
                kv_norm_n = kv_norm / (kv_norm.max() + 1e-8)
                pos_frac  = torch.linspace(0, 1, T)
                err4      = quantize_rel_error(kv, 4)
                err8      = quantize_rel_error(kv, 8)
                q4_local  = (1.0 - err4).clamp(0, 1)
                q8_local  = (1.0 - err8).clamp(0, 1)
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
    print(f"Cached {signals.shape[0]} tokens to {cache_path}")
    return signals, q_local


def train_controller(signals, q_local, epochs=10, lr=1e-3, batch_size=256, beta_override=2.0):
    print(f"\nTraining controller on {signals.shape[0]} tokens, {epochs} epochs...")
    ctrl = BinaryFPGAControllerPertok(input_dim=4, hidden_dim=64)
    opt  = torch.optim.Adam(ctrl.parameters(), lr=lr)

    N = signals.shape[0]
    tau_schedule = torch.linspace(2.0, 0.1, epochs)

    for ep in range(epochs):
        tau = tau_schedule[ep].item()
        idx = torch.randperm(N)
        ep_loss = ep_quality = ep_bits = ep_cost = 0.0
        n_batch = 0
        for start in range(0, N, batch_size):
            batch_sig = signals[idx[start:start+batch_size]]
            batch_q   = q_local[idx[start:start+batch_size]]
            opt.zero_grad()
            probs, _ = ctrl(batch_sig, tau=tau)
            costs = FPGA_COSTS
            q     = batch_q
            quality_loss = 1.0 - (probs * q).sum(dim=-1).mean()
            fpga_cost    = (probs * costs).sum(dim=-1).mean()
            fpga_norm    = fpga_cost / 1.01
            loss = 1.0 * quality_loss + beta_override * fpga_norm
            avg_bits_b = (probs * torch.tensor([4., 8.])).sum(dim=-1).mean()
            loss.backward()
            opt.step()
            ep_loss    += loss.item()
            ep_quality += quality_loss.item()
            ep_bits    += avg_bits_b.item()
            ep_cost    += fpga_cost.item()
            n_batch    += 1
        if (ep + 1) % 3 == 0 or ep == epochs - 1:
            print(f"  ep {ep+1:2d}/{epochs} tau={tau:.2f} loss={ep_loss/n_batch:.4f} "
                  f"avg_bits={ep_bits/n_batch:.2f} fpga_cost={ep_cost/n_batch:.3f}")
    return ctrl


def eval_bit_distribution(ctrl, signals, q_local):
    bits_pred = ctrl.predict_bits(signals)
    from collections import Counter
    c = Counter(bits_pred)
    total = len(bits_pred)
    bit_dist = {str(b): round(100.0 * c.get(b, 0) / total, 1) for b in BIT_CLASSES}
    avg_bits = sum(b * c.get(b, 0) for b in BIT_CLASSES) / total
    p4 = c.get(4, 0) / total
    p8 = c.get(8, 0) / total
    fpga_cost = p4 * 0.290 + p8 * 0.560
    fpga_speedup = 1.010 / fpga_cost if fpga_cost > 0 else 0
    avg_q4 = q_local[:, 0].mean().item()
    avg_q8 = q_local[:, 1].mean().item()
    return {
        "bit_dist": bit_dist,
        "avg_bits": round(avg_bits, 3),
        "fpga_cost": round(fpga_cost, 4),
        "fpga_speedup": round(fpga_speedup, 3),
        "avg_q4_local": round(avg_q4, 4),
        "avg_q8_local": round(avg_q8, 4),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("Smoke test: per-token pertok pipeline on SmolLM-360M (CPU)")
    print("Expected: ~100% 4-bit (eff_residual=8.1% < threshold)")
    print("=" * 60)

    device = "cpu"
    signals, q_local = extract_signals(MODEL_NAME, TRAIN_TEXTS, CACHE_PATH, device)
    print(f"Signals: {signals.shape}, q_local: {q_local.shape}")
    print(f"avg q4_local={q_local[:,0].mean():.4f}, avg q8_local={q_local[:,1].mean():.4f}")

    # Sweep betas to find the mixed-allocation regime at 360M
    # (at 360M, eff_residual=8.1% so we expect high 4-bit for all sufficiently large betas)
    import numpy as np
    q4_np = q_local[:, 0].numpy()
    q8_np = q_local[:, 1].numpy()
    gap = q8_np - q4_np
    print(f"\nq8-q4 gap: mean={gap.mean():.4f} std={gap.std():.4f}")
    for beta in [0.5, 1.0, 1.5, 2.0, 3.0]:
        thr = beta * 0.270 / 1.01
        f4 = (gap < thr).mean()
        print(f"  beta={beta}: threshold={thr:.4f} -> est {f4*100:.1f}% 4-bit")

    print("\nRunning beta sweep [1.0, 1.5, 2.0, 3.0]...")
    best_ctrl, best_metrics, best_beta = None, None, None
    for beta in [1.0, 1.5, 2.0, 3.0]:
        ctrl = train_controller(signals, q_local, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, beta_override=beta)
        m = eval_bit_distribution(ctrl, signals, q_local)
        p4 = m["bit_dist"].get("4", 0)
        print(f"  beta={beta}: {m['bit_dist']} -> speedup={m['fpga_speedup']:.2f}x")
        # Pick mixed or mostly-4-bit with highest speedup
        if best_metrics is None or m["fpga_speedup"] > best_metrics["fpga_speedup"]:
            best_ctrl, best_metrics, best_beta = ctrl, m, beta
    metrics = best_metrics
    print(f"\nBest beta: {best_beta}")

    elapsed = time.time() - t0
    result = {
        "model": MODEL_NAME,
        "type": "smoke_test_360m_pertok",
        "elapsed_s": round(elapsed, 1),
        **metrics,
    }

    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"  Bit distribution: {metrics['bit_dist']}")
    print(f"  avg_bits:         {metrics['avg_bits']:.2f}")
    print(f"  FPGA cost:        {metrics['fpga_cost']:.4f}")
    print(f"  FPGA speedup:     {metrics['fpga_speedup']:.2f}x")
    print(f"  avg q4_local:     {metrics['avg_q4_local']:.4f}")
    print(f"  avg q8_local:     {metrics['avg_q8_local']:.4f}")
    print(f"  Elapsed:          {elapsed:.1f}s")

    if metrics["bit_dist"].get("4", 0) >= 90:
        print("\n[PASS] 360M converges to 100% 4-bit as expected (eff_residual=8.1%)")
    else:
        print("\n[NOTE] Mixed allocation at 360M — unexpected, check quality signals")

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {RESULT_PATH}")
    return result


if __name__ == "__main__":
    main()
