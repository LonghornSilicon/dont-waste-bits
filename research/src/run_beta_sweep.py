"""
H3 beta-sweep experiment: does increasing the latency penalty beta
drive avg_bits toward paper's 5.05 while maintaining accuracy?

Tests beta in [0.1, 0.5, 1.0, 2.0] with 100 train samples, 5 epochs,
100 eval samples. All using the saved signal extraction (reuses signals
from training to save time).

Protocol (locked before running):
- Hypothesis: Higher beta -> lower avg_bits, possibly lower accuracy
  - If accuracy stays ~37% at avg_bits ~5: dual-objective tension is β-tunable
  - If accuracy collapses with lower bits: confirms compound loss alone insufficient
  - Paper target: 41.2% at 5.05 bits
"""

import torch
import torch.nn.functional as F
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

BETAS = [0.1, 0.5, 1.0, 2.0]
TRAIN_SAMPLES = 100
EVAL_SAMPLES = 100
EPOCHS = 5
MODEL_ID = "HuggingFaceTB/SmolLM-360M"
OUTPUT_DIR = Path("research/data")


def run_beta(model, tokenizer, train_texts, eval_ds, beta, device="cpu"):
    from dwb_implementation import DWBController, DWBLoss, build_training_dataset
    from eval_dwb import predict_bit_widths, score_continuation_dwb, make_per_token_hook
    from transformers import AutoModelForCausalLM

    print(f"\n  === beta={beta} ===", flush=True)

    # Build training data
    signals, targets = build_training_dataset(model, tokenizer, train_texts, device)
    print(f"  {len(signals)} token samples", flush=True)

    n = len(signals)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    tr_idx, val_idx = idx[:split], idx[split:]

    controller = DWBController().to(device)
    loss_fn = DWBLoss(alpha=1.0, beta=beta, gamma=0.1).to(device)
    opt = torch.optim.Adam(controller.parameters(), lr=3e-3)

    best_val = 0.0
    for epoch in range(EPOCHS):
        controller.train()
        perm = torch.randperm(len(tr_idx))
        for i in range(0, len(tr_idx), 256):
            batch = tr_idx[perm[i:i+256]]
            logits = controller(signals[batch].to(device))
            loss = loss_fn(logits, targets[batch].to(device))
            opt.zero_grad(); loss.backward(); opt.step()

        controller.eval()
        with torch.no_grad():
            val_logits = controller(signals[val_idx].to(device))
            val_acc = (val_logits.argmax(dim=-1) == targets[val_idx].to(device)).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
        print(f"  Epoch {epoch+1}/{EPOCHS}: val_acc={val_acc:.3f}", flush=True)

    print(f"  Best val_acc={best_val:.3f}", flush=True)

    # Evaluate on HellaSwag
    print(f"  Evaluating {EVAL_SAMPLES} samples...", flush=True)
    freq_table = Counter()
    for ex in eval_ds:
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        freq_table.update(tokenizer.encode(ctx))

    # Reload eager model for signal extraction
    eager_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    eager_model.eval()

    from eval_dwb import extract_signals_for_sequence
    correct = 0
    bit_dist = Counter()
    t0 = time.time()

    for i, ex in enumerate(eval_ds):
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        sig, _ = extract_signals_for_sequence(eager_model, tokenizer, ctx, freq_table, device)
        bit_widths = predict_bit_widths(controller, sig, device)
        for b in bit_widths:
            bit_dist[b] += 1

        scores = [
            score_continuation_dwb(model, tokenizer, ctx, " " + e, bit_widths, device)
            for e in ex["endings"]
        ]
        if max(range(4), key=lambda j: scores[j]) == int(ex["label"]):
            correct += 1

        if (i+1) % 25 == 0:
            print(f"  [{i+1}/{EVAL_SAMPLES}] acc={correct/(i+1)*100:.1f}%", flush=True)

    del eager_model

    acc = correct / EVAL_SAMPLES * 100
    total_toks = sum(bit_dist.values())
    avg_bits = sum(b * c for b, c in bit_dist.items()) / max(1, total_toks)
    bit_pct = {b: round(c / total_toks * 100, 1) for b, c in sorted(bit_dist.items())}

    print(f"  RESULT: beta={beta} -> acc={acc:.1f}%, avg_bits={avg_bits:.2f}, val_acc={best_val:.3f}", flush=True)
    print(f"  Bit dist: {bit_pct}", flush=True)

    return {
        "beta": beta, "accuracy": acc, "avg_bits": avg_bits,
        "val_acc": best_val, "bit_dist_pct": bit_pct,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    print(f"Loading HellaSwag...", flush=True)
    train_ds = load_dataset("Rowan/hellaswag", split="train").select(range(TRAIN_SAMPLES))
    eval_ds = load_dataset("Rowan/hellaswag", split="validation").select(range(EVAL_SAMPLES))

    train_texts = [
        ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        for ex in train_ds
    ]

    results = []
    for beta in BETAS:
        r = run_beta(model, tokenizer, train_texts, eval_ds, beta)
        results.append(r)
        print(f"\n  Cumulative: {[(r['beta'], r['accuracy'], r['avg_bits']) for r in results]}", flush=True)

    print("\n=== BETA SWEEP COMPLETE ===", flush=True)
    print(f"{'Beta':>6} | {'Acc':>6} | {'avg_bits':>8} | {'val_acc':>8}", flush=True)
    print("-" * 40, flush=True)
    for r in results:
        marker = " <-- paper bits" if abs(r['avg_bits'] - 5.05) < 1.0 else ""
        print(f"{r['beta']:>6.1f} | {r['accuracy']:>5.1f}% | {r['avg_bits']:>8.2f} | {r['val_acc']:>8.3f}{marker}", flush=True)
    print(f"\nPaper target: 41.2% @ 5.05 avg_bits", flush=True)

    out = {
        "experiment": "beta_sweep_h3",
        "date": datetime.now().isoformat(),
        "model": "smollm-360m",
        "train_samples": TRAIN_SAMPLES,
        "eval_samples": EVAL_SAMPLES,
        "epochs": EPOCHS,
        "paper_target": {"accuracy": 41.2, "avg_bits": 5.05},
        "results": results,
    }
    fname = OUTPUT_DIR / f"beta_sweep_{datetime.now():%Y%m%d_%H%M}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {fname}", flush=True)


if __name__ == "__main__":
    main()
