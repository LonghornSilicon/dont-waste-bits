"""
Fine beta sweep for OPT-125M around predicted beta*=0.798.
Uses cached signals. Sweeps beta in [0.3, 1.5] to bracket the transition.
"""
import json, time
from pathlib import Path
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

CACHE_PATH  = Path(__file__).parent.parent / "results" / "cross_arch_opt125m_cache.pt"
RESULT_PATH = Path(__file__).parent.parent / "results" / "cross_arch_opt125m_fine.json"
BIT_CLASSES = [4, 8]
FPGA_COSTS  = torch.tensor([0.29, 0.56], dtype=torch.float32)
EPOCHS, LR, BATCH_SZ = 10, 1e-3, 256
BETAS = [0.3, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.2, 1.5]
N_SEEDS = 3
PREDICTED_BETA_STAR = 0.798


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
            return [BIT_CLASSES[i] for i in self.net(x).argmax(dim=-1).tolist()]


def train_and_eval(signals, q_local, beta, seed=0):
    torch.manual_seed(seed)
    ctrl = BinaryFPGAController()
    opt  = torch.optim.Adam(ctrl.parameters(), lr=LR)
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
            loss = (1.0 - (probs * bq).sum(-1).mean()) + beta * (probs * FPGA_COSTS).sum(-1).mean() / 1.01
            loss.backward()
            opt.step()
    bits = ctrl.predict_bits(signals)
    c = Counter(bits)
    p4 = round(100.0 * c.get(4, 0) / len(bits), 1)
    fpga = p4/100 * 0.290 + (1-p4/100) * 0.560
    return p4, round(1.010/fpga, 3)


def main():
    if not CACHE_PATH.exists():
        print("ERROR: run cross_arch_beta_cal.py first")
        return

    d = torch.load(CACHE_PATH, weights_only=True)
    sigs, qs = d["signals"], d["q_local"]
    gap = (qs[:, 1] - qs[:, 0]).numpy()
    gap_mean = float(gap.mean())
    print(f"OPT-125M: {sigs.shape[0]} tokens  gap_mean={gap_mean:.4f}  "
          f"predicted beta*={PREDICTED_BETA_STAR}")
    print(f"Sweeping {len(BETAS)} betas x {N_SEEDS} seeds...\n")

    t0 = time.time()
    results = {}
    for beta in BETAS:
        p4s = [train_and_eval(sigs, qs, beta, seed=s)[0] for s in range(N_SEEDS)]
        mean_p4 = float(np.mean(p4s))
        fpga = mean_p4/100 * 0.290 + (1-mean_p4/100) * 0.560
        speedup = round(1.010/fpga, 3)
        threshold = round(beta * 0.267, 4)
        regime = "8-bit" if mean_p4 < 5 else ("4-bit" if mean_p4 > 95 else "MIXED")
        print(f"  beta={beta:.2f}  threshold={threshold:.4f}  "
              f"p4={mean_p4:.1f}%  speedup={speedup:.2f}x  [{regime}]")
        results[str(beta)] = {
            "beta": beta, "threshold": threshold,
            "p4_runs": p4s, "p4_mean": round(mean_p4, 1), "speedup": speedup,
            "regime": regime,
        }

    # Find transition window
    sorted_betas = sorted(results.keys(), key=float)
    transition_lo = transition_hi = None
    for i, bk in enumerate(sorted_betas[:-1]):
        r0 = results[bk]["p4_mean"]
        r1 = results[sorted_betas[i+1]]["p4_mean"]
        if r0 < 5 and r1 >= 5:
            transition_lo = float(bk)
            transition_hi = float(sorted_betas[i+1])

    print(f"\nPredicted beta*={PREDICTED_BETA_STAR:.3f}")
    if transition_lo:
        print(f"Measured transition: [{transition_lo}, {transition_hi}]")
        mid = (transition_lo + transition_hi) / 2
        error = abs(PREDICTED_BETA_STAR - mid)
        print(f"Error vs theory: {error:.3f}  (prediction {'CONFIRMED' if error < 0.1 else 'REFUTED'})")
    else:
        print("No transition found in sweep range")
        error = None

    output = {
        "model": "facebook/opt-125m",
        "gap_mean": round(gap_mean, 4),
        "predicted_beta_star": PREDICTED_BETA_STAR,
        "measured_transition": [transition_lo, transition_hi] if transition_lo else None,
        "theory_error": round(error, 3) if error else None,
        "formula_confirmed": error < 0.1 if error else False,
        "elapsed_s": round(time.time()-t0, 1),
        "sweep": results,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {RESULT_PATH}")
    return output


if __name__ == "__main__":
    main()
