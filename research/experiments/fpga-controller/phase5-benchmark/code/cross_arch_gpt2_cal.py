"""
GPT-2 (OpenAI, 124M) cross-architecture beta* calibration.
Hypothesis: beta* = gap_mean/0.267 is hardware-universal across GPT-2 architecture.

GPT-2 uses Conv1D(c_attn) for combined QKV — different from LLaMA (k_proj/v_proj)
and OPT (self_attn.k_proj/v_proj). We hook c_attn and split to extract K, V.

Prediction: gap_mean drives beta* regardless of model family.
OPT-125M had gap_mean=0.213 (768 dim). GPT-2 also has 768 dim — predict similar
but potentially different due to different pre-training data and MLP structure.
"""
import json
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CACHE_PATH = Path(__file__).parent.parent / "results" / "cross_arch_gpt2_cache.pt"
RESULT_PATH = Path(__file__).parent.parent / "results" / "cross_arch_gpt2_fine.json"
BIT_CLASSES = [4, 8]
FPGA_COSTS = torch.tensor([0.29, 0.56], dtype=torch.float32)
N_TEXTS = 10
MAX_LEN = 64
EPOCHS, LR, BATCH_SZ = 10, 1e-3, 256
BETAS = [0.3, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.2, 1.5]
N_SEEDS = 3


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

    print("Loading GPT-2 (124M)...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()

    hidden_dim = model.config.n_embd  # 768
    n_layers = model.config.n_layer   # 12
    print(f"GPT-2: {n_layers} layers, hidden_dim={hidden_dim}, "
          f"n_heads={model.config.n_head}")

    all_q4, all_q8 = [], []
    layer_kv = {i: {"k": [], "v": []} for i in range(n_layers)}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output: [batch, seq_len, 3*hidden_dim]
            k = output[:, :, hidden_dim:2*hidden_dim].detach()
            v = output[:, :, 2*hidden_dim:].detach()
            layer_kv[layer_idx]["k"].append(k)
            layer_kv[layer_idx]["v"].append(v)
        return hook

    hooks = []
    for i in range(n_layers):
        h = model.transformer.h[i].attn.c_attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    print(f"Loading WikiText-2 ({N_TEXTS} texts, max_len={MAX_LEN})...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                      trust_remote_code=True)
    texts = [r["text"] for r in ds if len(r["text"].split()) > 20][:N_TEXTS]

    t0 = time.time()
    with torch.no_grad():
        for text in texts:
            enc = tok(text, return_tensors="pt", max_length=MAX_LEN, truncation=True)
            for i in range(n_layers):
                layer_kv[i]["k"].clear()
                layer_kv[i]["v"].clear()
            model(**enc)
            for i in range(n_layers):
                if not layer_kv[i]["k"]:
                    continue
                k = layer_kv[i]["k"][0][0]  # [seq_len, hidden]
                v = layer_kv[i]["v"][0][0]
                for t in range(k.shape[0]):
                    q4 = quant_quality(k[t], 4)
                    q8 = quant_quality(k[t], 8)
                    all_q4.append(float(q4))
                    all_q8.append(float(q8))
                    q4 = quant_quality(v[t], 4)
                    q8 = quant_quality(v[t], 8)
                    all_q4.append(float(q4))
                    all_q8.append(float(q8))

    for h in hooks:
        h.remove()

    q4t = torch.tensor(all_q4, dtype=torch.float32)
    q8t = torch.tensor(all_q8, dtype=torch.float32)
    gap = (q8t - q4t).numpy()
    gap_mean = float(gap.mean())
    gap_std = float(gap.std())
    n_tokens = len(all_q4)

    print(f"Extracted {n_tokens} token signals in {time.time()-t0:.1f}s")
    print(f"gap_mean={gap_mean:.4f}  gap_std={gap_std:.4f}")
    print(f"Predicted beta* = {gap_mean:.4f}/0.267 = {gap_mean/0.267:.3f}")

    q_local = torch.stack([q4t, q8t], dim=1)
    signals = q_local.clone()

    torch.save({"signals": signals, "q_local": q_local,
                "gap_mean": gap_mean, "gap_std": gap_std,
                "n_tokens": n_tokens, "model": "gpt2"},
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
    print(f"\nSweeping {len(BETAS)} betas x {N_SEEDS} seeds "
          f"(predicted beta*={predicted_beta_star:.3f})...\n")
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
        print(f"  beta={beta:.2f}  threshold={threshold:.4f}  "
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

    print(f"\nPredicted beta*={predicted_beta_star:.3f}")
    error = None
    if transition_lo is not None:
        mid = (transition_lo + transition_hi) / 2
        error = abs(predicted_beta_star - mid)
        confirmed = error < 0.1
        print(f"Measured transition: [{transition_lo}, {transition_hi}]")
        print(f"Error vs theory: {error:.3f}  "
              f"({'CONFIRMED' if confirmed else 'REFUTED'})")
    else:
        print("No transition found in sweep range")

    output = {
        "model": "openai/gpt2 (GPT-2 Small, 124M)",
        "architecture": "GPT-2 (OpenAI) — Conv1D combined QKV",
        "gap_mean": round(gap_mean, 4),
        "predicted_beta_star": round(predicted_beta_star, 3),
        "measured_transition": ([transition_lo, transition_hi]
                                if transition_lo is not None else None),
        "theory_error": round(error, 3) if error is not None else None,
        "formula_confirmed": (error < 0.1) if error is not None else False,
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
        print(f"GPT-2: {signals.shape[0]} tokens  gap_mean={gap_mean:.4f}  "
              f"predicted beta*={gap_mean/0.267:.3f}")
    else:
        signals, q_local, gap_mean = extract_kv_signals()

    run_sweep(signals, q_local, gap_mean)


if __name__ == "__main__":
    main()
