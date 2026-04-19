"""
H4: Cross-model validation on SmolLM-135M.
Paper Table 3: FP16=37.2%, Static-4bit=33.6%, DWB=36.9%
"""
import sys, json, time, torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from kv_cache_quant import attach_kv_hooks
from eval_hellaswag import score_continuation

MODEL_ID = "HuggingFaceTB/SmolLM-135M"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
DEVICE = "cpu"

PAPER = {"fp16": 37.20, "static4bit": 33.60}
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
        if i > 0 and i % 20 == 0:
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

out = {"model": "smollm-135m", "limit": LIMIT, "metric": "acc_unnorm",
       "date": datetime.now().isoformat(), "paper": PAPER, "conditions": {}}

for cond, mode in CONDITIONS:
    print(f"\n--- {cond} ---", flush=True)
    r = evaluate(model, tokenizer, ds, mode)
    paper_key = "fp16" if cond == "fp16" else "static4bit"
    r["paper_target"] = PAPER.get(paper_key)
    r["delta"] = round(r["accuracy"] - PAPER[paper_key], 2)
    out["conditions"][cond] = r
    print(f"  {cond}: {r['accuracy']:.2f}%  paper: {r['paper_target']}%  delta: {r['delta']:+.2f}pp", flush=True)

out_dir = Path("research/data")
fname = out_dir / f"h4_smollm135m_{LIMIT}samp_{datetime.now():%Y%m%d_%H%M}.json"
with open(fname, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {fname}", flush=True)

print("\n=== H4 RESULTS: SmolLM-135M vs Paper Table 3 ===")
for cond, r in out["conditions"].items():
    print(f"  {cond}: {r['accuracy']:.1f}%  paper: {r.get('paper_target','—')}%  delta: {r['delta']:+.1f}pp")
