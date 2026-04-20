"""
Fine-grained beta sweep near beta*=1.584 for SmolLM-1.7B.
Uses cached signals from beta_calibration_1b7.py (no model loading needed).
Protocol: prediction is transition at beta*=1.584 (gap_mean/0.267=0.4235/0.267).
Expected: beta=1.58 near 50% split; transition window [1.57, 1.60].
"""
import json
import sys
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_PATH  = Path(__file__).parent.parent / "results" / "beta_cal_1b7_cache.pt"
RESULT_PATH = Path(__file__).parent.parent / "results" / "beta_transition_fine_1b7.json"
OUTPUT_DIR  = RESULT_PATH.parent
OUTPUT_DIR.mkdir(exist_ok=True)

BIT_CLASSES = [4, 8]
FPGA_COSTS  = torch.tensor([0.29, 0.56], dtype=torch.float32)
EPOCHS      = 10
LR          = 1e-3
BATCH_SIZE  = 256


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
    if not CACHE_PATH.exists():
        print(f"ERROR: cache not found at {CACHE_PATH}")
        print("Run beta_calibration_1b7.py first to generate the cache.")
        return

    print("=" * 60)
    print("Fine beta sweep: SmolLM-1.7B (using cached signals)")
    print(f"Gap mean=0.4235, predicted beta*=1.584")
    print("=" * 60)

    d = torch.load(CACHE_PATH, weights_only=True)
    signals, q_local = d["signals"], d["q_local"]
    gap = (q_local[:, 1] - q_local[:, 0]).numpy()
    gap_mean = float(gap.mean())
    print(f"Loaded {signals.shape[0]} tokens. gap_mean={gap_mean:.4f}")

    # Fine sweep bracketing the predicted transition at 1.584
    betas = [1.50, 1.52, 1.55, 1.57, 1.58, 1.59, 1.60, 1.62, 1.65, 1.70]

    t0 = time.time()
    results = []
    for beta in betas:
        thr = beta * 0.270 / 1.01
        m = train_and_eval(signals, q_local, beta)
        if m["p4_pct"] >= 90:
            outcome = "4-bit dominant"
        elif m["p4_pct"] <= 10:
            outcome = "8-bit dominant"
        else:
            outcome = f"mixed (near transition)"
        print(f"  beta={beta:.2f}: thr={thr:.4f} -> p4={m['p4_pct']}% speedup={m['fpga_speedup']}x [{outcome}]")
        results.append({"beta": beta, "threshold": round(thr, 4), **m, "outcome": outcome})

    elapsed = time.time() - t0
    output = {
        "model": "HuggingFaceTB/SmolLM-1.7B",
        "n_tokens": int(signals.shape[0]),
        "gap_mean": round(gap_mean, 4),
        "predicted_transition_beta": round(gap_mean / 0.267, 3),
        "elapsed_s": round(elapsed, 1),
        "sweep": results,
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {RESULT_PATH} ({elapsed:.0f}s)")

    # Find transition window
    below = [r for r in results if r["p4_pct"] <= 10]
    above = [r for r in results if r["p4_pct"] >= 90]
    if below and above:
        beta_lo = max(r["beta"] for r in below)
        beta_hi = min(r["beta"] for r in above)
        print(f"\nTransition window: [{beta_lo}, {beta_hi}] (predicted beta*={output['predicted_transition_beta']})")
    return output


if __name__ == "__main__":
    main()
