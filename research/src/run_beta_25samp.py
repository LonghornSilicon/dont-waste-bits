"""
Minimal beta sweep for betas 1.0 and 2.0 with only 25 training samples.
Phase 1 extracts signals for 25 samples (confirmed to succeed on this machine).
Phase 2 trains controller with NO model in memory.
Goal: get val_acc and avg_bits only (trend confirmation).
"""
import torch, json, gc, sys
from pathlib import Path
from datetime import datetime
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))

BETAS = [1.0, 2.0]
TRAIN_SAMPLES = 25
EPOCHS = 5
MODEL_ID = "HuggingFaceTB/SmolLM-360M"
OUTPUT_DIR = Path("research/data")
CACHE = OUTPUT_DIR / "beta_sweep_25samp_signals.pt"


def phase1(tokenizer, device="cpu"):
    from eval_dwb import extract_signals_for_sequence
    from transformers import AutoModelForCausalLM
    from datasets import load_dataset
    print("Phase1: loading eager model...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, attn_implementation="eager").to(device)
    m.eval()
    ds = load_dataset("Rowan/hellaswag", split="train").select(range(TRAIN_SAMPLES))
    texts = [ex["activity_label"]+": "+ex["ctx_a"]+" "+ex["ctx_b"].capitalize() for ex in ds]
    fc = Counter()
    for t in texts: fc.update(tokenizer.encode(t))
    sigs, tgts = [], []
    for i, t in enumerate(texts):
        s, g = extract_signals_for_sequence(m, tokenizer, t, fc, device)
        sigs.append(s); tgts.append(g)
        print(f"  {i+1}/{TRAIN_SAMPLES}", flush=True)
    del m; gc.collect()
    sig = torch.cat(sigs); tgt = torch.cat(tgts)
    torch.save({"signals": sig, "targets": tgt}, CACHE)
    print(f"Phase1 done: {len(sig)} tokens saved.", flush=True)
    return sig, tgt


def phase2(signals, targets, beta, device="cpu"):
    from dwb_implementation import DWBController, DWBLoss
    from eval_dwb import predict_bit_widths
    print(f"\nPhase2: beta={beta}", flush=True)
    n = len(signals)
    idx = torch.randperm(n)
    split = int(0.8*n)
    tr, val = idx[:split], idx[split:]
    ctrl = DWBController().to(device)
    lf = DWBLoss(1.0, beta, 0.1).to(device)
    opt = torch.optim.Adam(ctrl.parameters(), lr=3e-3)
    best = 0.0
    for ep in range(EPOCHS):
        ctrl.train()
        p = torch.randperm(len(tr))
        for i in range(0, len(tr), 64):
            b = tr[p[i:i+64]]
            loss = lf(ctrl(signals[b].to(device)), targets[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        ctrl.eval()
        with torch.no_grad():
            va = (ctrl(signals[val].to(device)).argmax(-1) == targets[val].to(device)).float().mean().item()
        if va > best: best = va
        print(f"  ep{ep+1}: val_acc={va:.3f}", flush=True)
    ctrl.eval()
    with torch.no_grad():
        bits = predict_bit_widths(ctrl, signals, device)
    bd = Counter(bits)
    tot = sum(bd.values())
    avg = sum(b*c for b,c in bd.items())/max(1,tot)
    pct = {b: round(c/tot*100,1) for b,c in sorted(bd.items())}
    print(f"  RESULT: beta={beta}, val_acc={best:.3f}, avg_bits={avg:.2f}, bits={pct}", flush=True)
    return {"beta": beta, "accuracy": None, "avg_bits": avg, "val_acc": best, "bit_dist_pct": pct, "note": f"train-only 25samp"}


def main():
    from transformers import AutoTokenizer
    print("Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if CACHE.exists():
        print("Loading cached signals...", flush=True)
        c = torch.load(CACHE, weights_only=True)
        sigs, tgts = c["signals"], c["targets"]
    else:
        sigs, tgts = phase1(tok)

    partial = OUTPUT_DIR / "beta_sweep_partial.json"
    with open(partial) as f: results = json.load(f)
    done = {r["beta"] for r in results}

    for beta in BETAS:
        if beta in done:
            print(f"  Skip beta={beta}", flush=True); continue
        r = phase2(sigs, tgts, beta)
        results.append(r)
        with open(partial, "w") as f: json.dump(results, f, indent=2)
        print(f"  Saved ({len(results)} betas)", flush=True)

    print("\n=== DONE ===", flush=True)
    for r in results:
        print(f"  beta={r['beta']}: acc={r['accuracy']}, avg_bits={r['avg_bits']:.2f}, val_acc={r['val_acc']:.3f}  {r.get('note','')}", flush=True)
    fname = OUTPUT_DIR / f"beta_sweep_{datetime.now():%Y%m%d_%H%M}.json"
    with open(fname, "w") as f: json.dump({"experiment":"beta_sweep_h3","date":datetime.now().isoformat(),"results":results}, f, indent=2)
    print(f"Saved: {fname}", flush=True)

if __name__ == "__main__": main()
