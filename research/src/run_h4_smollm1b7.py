"""
H4 extension: Cross-model validation on SmolLM-1.7B.
Paper Table 3: FP16=49.0%, Static-4bit=41.1%, DWB=48.6%

Key question: does standard INT4 remain lossless at 1.7B, or does
the paper's 8pp gap indicate genuine degradation at scale?
"""
import sys, json, time, torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from kv_cache_quant import attach_kv_hooks
from eval_hellaswag import score_continuation

MODEL_ID = "HuggingFaceTB/SmolLM-1.7B"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 50
DEVICE = "cpu"

PAPER = {"fp16": 49.00, "static4bit": 41.10, "dwb": 48.60}
CONDITIONS = [
    ("fp16", None),
    ("static4bit_pertensor", "static4bit"),
    ("int4_int3range", "int4_int3range"),
]


def evaluate(model, tokenizer, ds, quant_mode=None):
    handles = []
    if quant_mode:
        handles = attach_kv_hooks(model, quant_mode)
    correct = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        if i > 0 and i % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(ds) - i)
            print(f"  [{i}/{LIMIT}] acc={correct/i*100:.1f}% eta={eta:.0f}s", flush=True)
        ctx = ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
        scores = [score_continuation(model, tokenizer, ctx, " " + e) for e in ex["endings"]]
        if max(range(4), key=lambda j: scores[j]) == int(ex["label"]):
            correct += 1
    for h in handles:
        h.remove()
    elapsed = time.time() - t0
    acc = correct / len(ds) * 100
    print(f"  Result: {correct}/{len(ds)} = {acc:.2f}%  ({elapsed:.0f}s)", flush=True)
    return {"accuracy": acc, "correct": correct, "total": len(ds), "elapsed_s": round(elapsed, 1)}


print(f"Loading {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

print(f"Loading HellaSwag ({LIMIT} samples)...", flush=True)
ds = load_dataset("Rowan/hellaswag", split="validation").select(range(LIMIT))

out = {"model": "smollm-1.7b", "limit": LIMIT, "metric": "acc_unnorm",
       "date": datetime.now().isoformat(), "paper": PAPER, "conditions": {}}

for cond, mode in CONDITIONS:
    print(f"\n--- {cond} ---", flush=True)
    result = evaluate(model, tokenizer, ds, quant_mode=mode)
    out["conditions"][cond] = result

ts = datetime.now().strftime("%Y%m%d_%H%M")
fname = f"research/data/h4_1b7_{LIMIT}samp_{ts}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {fname}", flush=True)

print("\n=== H4 SmolLM-1.7B RESULTS ===")
for cond, res in out["conditions"].items():
    paper_key = "fp16" if cond == "fp16" else ("static4bit" if "int4" in cond.lower() and "int3" not in cond.lower() else "static4bit")
    paper_val = PAPER.get(paper_key, "—")
    delta = res["accuracy"] - paper_val if isinstance(paper_val, float) else 0
    print(f"  {cond}: {res['accuracy']:.1f}%  (paper: {paper_val}%, delta: {delta:+.1f}pp)")
