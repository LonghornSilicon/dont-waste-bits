"""
Fine-grained beta sweep around the predicted phase transition at 360M.
Theory: transition at beta = gap_mean / 0.267 = 0.337/0.267 = 1.26.
Tests beta in [1.1, 1.2, 1.25, 1.3, 1.4] using cached signals.
"""

import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE_PATH  = Path(__file__).parent.parent / "results" / "smoke_360m_pertok_cache.pt"
RESULT_PATH = Path(__file__).parent.parent / "results" / "beta_transition_fine.json"
BIT_CLASSES = [4, 8]
FPGA_COSTS  = torch.tensor([0.29, 0.56], dtype=torch.float32)
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 256


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


def train_controller(signals, q_local, beta, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    ctrl = BinaryFPGAControllerPertok(input_dim=4, hidden_dim=64)
    opt  = torch.optim.Adam(ctrl.parameters(), lr=lr)
    N = signals.shape[0]
    tau_schedule = torch.linspace(2.0, 0.1, epochs)
    for ep in range(epochs):
        tau = tau_schedule[ep].item()
        idx = torch.randperm(N)
        for start in range(0, N, batch_size):
            batch_sig = signals[idx[start:start+batch_size]]
            batch_q   = q_local[idx[start:start+batch_size]]
            opt.zero_grad()
            probs, _ = ctrl(batch_sig, tau=tau)
            quality_loss = 1.0 - (probs * batch_q).sum(dim=-1).mean()
            fpga_cost    = (probs * FPGA_COSTS).sum(dim=-1).mean()
            fpga_norm    = fpga_cost / 1.01
            loss = 1.0 * quality_loss + beta * fpga_norm
            loss.backward()
            opt.step()
    return ctrl


def eval_controller(ctrl, signals):
    bits_pred = ctrl.predict_bits(signals)
    from collections import Counter
    c = Counter(bits_pred)
    total = len(bits_pred)
    p4 = c.get(4, 0) / total
    p8 = c.get(8, 0) / total
    fpga_cost = p4 * 0.290 + p8 * 0.560
    speedup   = 1.010 / fpga_cost if fpga_cost > 0 else 0
    return {
        "p4_pct": round(100.0 * p4, 1),
        "p8_pct": round(100.0 * p8, 1),
        "avg_bits": round(4.0 * p4 + 8.0 * p8, 3),
        "fpga_cost": round(fpga_cost, 4),
        "fpga_speedup": round(speedup, 3),
    }


def main():
    if not CACHE_PATH.exists():
        print(f"Cache not found at {CACHE_PATH}. Run smoke_test_360m_pertok.py first.")
        return

    print("Loading cached signals...")
    data = torch.load(CACHE_PATH, weights_only=True)
    signals = data["signals"]
    q_local = data["q_local"]
    gap = (q_local[:, 1] - q_local[:, 0]).numpy()
    gap_mean = gap.mean()
    gap_std  = gap.std()
    print(f"Tokens: {signals.shape[0]}, gap mean={gap_mean:.4f} std={gap_std:.4f}")

    # Theoretical transition: beta * (C8-C4)/C_FP16 = gap_mean
    # C8-C4 = 0.560-0.290 = 0.270, C_FP16=1.010 -> effective = 0.270/1.010 = 0.2673
    pred_transition = gap_mean / 0.2673
    print(f"Predicted phase transition at beta = {pred_transition:.3f}")

    betas_to_test = [1.1, 1.2, 1.25, 1.3, 1.4]
    results = []

    t0 = time.time()
    for beta in betas_to_test:
        threshold = beta * 0.270 / 1.01
        print(f"\nbeta={beta} (threshold={threshold:.4f} vs gap_mean={gap_mean:.4f})...")
        ctrl = train_controller(signals, q_local, beta)
        m = eval_controller(ctrl, signals)
        outcome = "4-bit dominant" if m["p4_pct"] >= 90 else ("8-bit dominant" if m["p4_pct"] <= 10 else "mixed")
        print(f"  -> p4={m['p4_pct']}%, p8={m['p8_pct']}%, speedup={m['fpga_speedup']:.2f}x [{outcome}]")
        results.append({"beta": beta, "threshold": round(threshold, 4), **m, "outcome": outcome})

    elapsed = time.time() - t0
    output = {
        "experiment": "beta_transition_fine_360m",
        "gap_mean": round(float(gap_mean), 4),
        "gap_std":  round(float(gap_std), 4),
        "predicted_transition_beta": round(pred_transition, 3),
        "elapsed_s": round(elapsed, 1),
        "sweep": results,
    }

    print("\n=== FINE BETA SWEEP RESULTS ===")
    print(f"Predicted transition: beta = {pred_transition:.3f}")
    for r in results:
        marker = " <-- TRANSITION" if (
            r == results[0] and r["p4_pct"] < 90
        ) or (
            results.index(r) > 0 and results[results.index(r)-1]["p4_pct"] < 90 and r["p4_pct"] >= 90
        ) else ""
        print(f"  beta={r['beta']}: {r['p4_pct']}% 4-bit{marker}")

    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULT_PATH}")
    return output


if __name__ == "__main__":
    main()
