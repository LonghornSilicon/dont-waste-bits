"""
Two-phase beta sweep for betas 1.0 and 2.0.
Phase 1: Extract and SAVE signals (loads eager model, del's it)
Phase 2: Load saved signals, train controller for each beta (no model needed)
This way we never have a model + training data in memory at the same time.
"""
import torch
import json
import gc
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

BETAS = [1.0, 2.0]
TRAIN_SAMPLES = 100
EPOCHS = 5
MODEL_ID = "HuggingFaceTB/SmolLM-360M"
OUTPUT_DIR = Path("research/data")
SIGNALS_CACHE = OUTPUT_DIR / "beta_sweep_signals_cache.pt"


def phase1_extract_signals(train_texts, tokenizer, device="cpu"):
    """Load eager model, extract signals for all train texts, save, delete model."""
    from eval_dwb import extract_signals_for_sequence
    from transformers import AutoModelForCausalLM

    print("Phase 1: Loading eager model for signal extraction...", flush=True)
    eager = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    eager.eval()

    freq_table = Counter()
    for txt in train_texts:
        freq_table.update(tokenizer.encode(txt))

    sigs, tgts = [], []
    for i, txt in enumerate(train_texts):
        s, t = extract_signals_for_sequence(eager, tokenizer, txt, freq_table, device)
        sigs.append(s); tgts.append(t)
        if (i + 1) % 25 == 0:
            print(f"  Signal {i+1}/{TRAIN_SAMPLES}", flush=True)

    signals = torch.cat(sigs)
    targets = torch.cat(tgts)
    print(f"  {len(signals)} tokens. Saving signals cache...", flush=True)
    torch.save({"signals": signals, "targets": targets}, SIGNALS_CACHE)
    del eager, sigs, tgts; gc.collect()
    print("Phase 1 complete. Eager model freed.", flush=True)
    return signals, targets


def phase2_train(signals, targets, beta, device="cpu"):
    """Train controller purely on cached signals. Zero model memory."""
    from dwb_implementation import DWBController, DWBLoss
    from eval_dwb import predict_bit_widths

    print(f"\nPhase 2: Training beta={beta}...", flush=True)
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
            vl = controller(signals[val_idx].to(device))
            val_acc = (vl.argmax(-1) == targets[val_idx].to(device)).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
        print(f"  Epoch {epoch+1}/{EPOCHS}: val_acc={val_acc:.3f}", flush=True)

    controller.eval()
    with torch.no_grad():
        all_bits = predict_bit_widths(controller, signals, device)
    bit_dist = Counter(all_bits)
    total = sum(bit_dist.values())
    avg_bits = sum(b * c for b, c in bit_dist.items()) / max(1, total)
    bit_pct = {b: round(c / total * 100, 1) for b, c in sorted(bit_dist.items())}

    print(f"  beta={beta}: val_acc={best_val:.3f}, avg_bits={avg_bits:.2f}, bits={bit_pct}", flush=True)
    return {
        "beta": beta, "accuracy": None, "avg_bits": avg_bits,
        "val_acc": best_val, "bit_dist_pct": bit_pct,
        "note": "train-only (signals only, no eval due to RAM constraints)",
    }


def main():
    from transformers import AutoTokenizer
    from datasets import load_dataset

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading HellaSwag train...", flush=True)
    train_ds = load_dataset("Rowan/hellaswag", split="train").select(range(TRAIN_SAMPLES))
    train_texts = [
        ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        for ex in train_ds
    ]

    # Phase 1: extract or load signals
    if SIGNALS_CACHE.exists():
        print("Loading cached signals...", flush=True)
        cache = torch.load(SIGNALS_CACHE, weights_only=True)
        signals, targets = cache["signals"], cache["targets"]
        print(f"  {len(signals)} cached tokens", flush=True)
    else:
        signals, targets = phase1_extract_signals(train_texts, tokenizer)

    # Phase 2: train for each beta
    partial_path = OUTPUT_DIR / "beta_sweep_partial.json"
    with open(partial_path) as f:
        results = json.load(f)
    done_betas = {r["beta"] for r in results}

    for beta in BETAS:
        if beta in done_betas:
            print(f"  Skipping beta={beta} (done)", flush=True)
            continue
        r = phase2_train(signals, targets, beta)
        results.append(r)
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Partial saved ({len(results)} betas done)", flush=True)

    print("\n=== COMPLETE ===", flush=True)
    for r in results:
        acc_str = f"{r['accuracy']:.1f}%" if r['accuracy'] is not None else "N/A (train-only)"
        print(f"  beta={r['beta']}: acc={acc_str}, avg_bits={r['avg_bits']:.2f}, val_acc={r['val_acc']:.3f}", flush=True)

    out = {"experiment": "beta_sweep_h3", "date": datetime.now().isoformat(),
           "model": MODEL_ID, "results": results}
    fname = OUTPUT_DIR / f"beta_sweep_{datetime.now():%Y%m%d_%H%M}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {fname}", flush=True)


if __name__ == "__main__":
    main()
