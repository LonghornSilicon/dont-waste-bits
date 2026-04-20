"""
Multi-seed reproducibility test for SmolLM-1.7B controller at beta=1.65.
Protocol: run 5 independent training seeds, report mean ± std of 4-bit%.
Expected: variance ~5-10pp due to Gumbel-softmax training stochasticity.
"""
import json, time, sys
from pathlib import Path
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

CACHE_PATH = Path(__file__).parent.parent / "results" / "beta_cal_1b7_cache.pt"
RESULT_PATH = Path(__file__).parent.parent / "results" / "reproducibility_1b7.json"
BIT_CLASSES = [4, 8]
FPGA_COSTS  = torch.tensor([0.29, 0.56], dtype=torch.float32)
EPOCHS, LR, BATCH_SIZE = 10, 1e-3, 256
N_SEEDS = 5
BETAS   = [1.60, 1.65, 1.70]

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


def train_and_eval(signals, q_local, beta, seed):
    torch.manual_seed(seed)
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
    return round(100.0 * p4, 1)


def main():
    if not CACHE_PATH.exists():
        print("ERROR: run beta_calibration_1b7.py first")
        return
    d = torch.load(CACHE_PATH, weights_only=True)
    signals, q_local = d["signals"], d["q_local"]
    print(f"Loaded {signals.shape[0]} tokens")
    print(f"Running {N_SEEDS} seeds × {len(BETAS)} betas...")

    t0 = time.time()
    results = {}
    for beta in BETAS:
        p4_vals = [train_and_eval(signals, q_local, beta, seed=i) for i in range(N_SEEDS)]
        mean = float(np.mean(p4_vals))
        std  = float(np.std(p4_vals))
        fpga_cost = mean/100 * 0.290 + (1 - mean/100) * 0.560
        speedup = round(1.010 / fpga_cost, 3)
        print(f"  beta={beta}: p4_pct={p4_vals} -> mean={mean:.1f}% ± {std:.1f}pp | speedup={speedup:.2f}x")
        results[str(beta)] = {"p4_pct_runs": p4_vals, "mean": round(mean, 1), "std": round(std, 1),
                              "speedup_mean": speedup}

    elapsed = time.time() - t0
    output = {"model": "HuggingFaceTB/SmolLM-1.7B", "n_seeds": N_SEEDS, "betas": BETAS,
              "elapsed_s": round(elapsed, 1), "results": results}
    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULT_PATH} ({elapsed:.0f}s)")
    print("\nConclusion: training stochasticity is ~±Xpp. Paper should report range or note this.")
    return output

if __name__ == "__main__":
    main()
