"""
Calibration sensitivity analysis: how few texts suffice to estimate β*?

Protocol: re-use cached 1.7B signals (15,360 tokens from 10 texts ≈ 1536 tok/text).
For each corpus size n_texts ∈ {1, 2, 3, 5, 7, 10}, draw 10 random subsamples,
compute gap_mean = mean(q8_local - q4_local) and β* = gap_mean / 0.267.
Report mean ± std of gap_mean and β* across subsamples.

Validates the paper claim: "<1 minute CPU calibration eliminates per-model grid search."
"""
import json, time, numpy as np
from pathlib import Path
import torch

CACHE_PATH   = Path(__file__).parent.parent / "results" / "beta_cal_1b7_cache.pt"
RESULT_PATH  = Path(__file__).parent.parent / "results" / "calibration_sensitivity_1b7.json"
TOKENS_PER_TEXT = 1536   # approx tokens per text in cache (15360 / 10)
N_REPS       = 20        # random subsamples per corpus size
CORPUS_SIZES = [1, 2, 3, 5, 7, 10]
TRUE_BETA_STAR = 1.584   # from full 10-text calibration


def main():
    if not CACHE_PATH.exists():
        print("ERROR: run beta_calibration_1b7.py first to populate cache")
        return

    d = torch.load(CACHE_PATH, weights_only=True)
    signals, q_local = d["signals"], d["q_local"]
    N = signals.shape[0]
    gap_all = (q_local[:, 1] - q_local[:, 0]).numpy()  # q8 - q4

    print(f"Loaded {N} tokens (gap_all: mean={gap_all.mean():.4f}, std={gap_all.std():.4f})")
    print(f"True beta* (full corpus): {TRUE_BETA_STAR}")
    print(f"Running {N_REPS} subsamples × {len(CORPUS_SIZES)} sizes...\n")

    results = {}
    for n_texts in CORPUS_SIZES:
        n_tokens = min(n_texts * TOKENS_PER_TEXT, N)
        gap_means = []
        beta_stars = []
        for _ in range(N_REPS):
            idx = np.random.choice(N, size=n_tokens, replace=False)
            gm = gap_all[idx].mean()
            bs = gm / 0.267
            gap_means.append(float(gm))
            beta_stars.append(float(bs))

        gm_mean = float(np.mean(gap_means))
        gm_std  = float(np.std(gap_means))
        bs_mean = float(np.mean(beta_stars))
        bs_std  = float(np.std(beta_stars))
        bs_max_err = float(np.max(np.abs(np.array(beta_stars) - TRUE_BETA_STAR)))

        print(f"  n_texts={n_texts:2d} (~{n_tokens} tok):  "
              f"gap_mean={gm_mean:.4f}+-{gm_std:.4f}  "
              f"beta*={bs_mean:.3f}+-{bs_std:.3f}  "
              f"max_err={bs_max_err:.3f}")

        results[str(n_texts)] = {
            "n_texts": n_texts, "n_tokens": n_tokens,
            "gap_mean_mean": round(gm_mean, 4), "gap_mean_std": round(gm_std, 4),
            "beta_star_mean": round(bs_mean, 3), "beta_star_std": round(bs_std, 3),
            "beta_star_max_err": round(bs_max_err, 3),
        }

    output = {
        "model": "HuggingFaceTB/SmolLM-1.7B",
        "n_total_tokens": int(N),
        "n_reps": N_REPS,
        "true_beta_star": TRUE_BETA_STAR,
        "results": results,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULT_PATH}")

    # Quick verdict
    for n_texts in [1, 3, 5]:
        r = results[str(n_texts)]
        verdict = 'ACCEPTABLE (<0.05)' if r['beta_star_max_err'] < 0.05 else 'TOO HIGH (>=0.05)'
        print(f"n_texts={n_texts}: beta* max error vs true = {r['beta_star_max_err']:.3f} ({verdict})")

    return output


if __name__ == "__main__":
    main()
