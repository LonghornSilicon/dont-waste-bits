"""
Phase 7f: kv_norm_inverted sanity check on SmolLM-1.7B (HellaSwag n=200)
========================================================================
Pathological reverse of Phase 7c's kv_norm strategy:
  - kv_norm          : bottom-74% by L2 norm → 4-bit  (sensible; high-norm tokens get 8-bit)
  - kv_norm_inverted : TOP-74% by L2 norm → 4-bit  (pathological; the precision goes to
                                                    small-norm tokens, starving the
                                                    informative high-norm ones)

Prediction: significant accuracy drop vs Phase 7c's forward kv_norm — confirms that the
forward direction isn't arbitrary. At a fixed 74/26 split, which tokens get the bits
matters only when the *worst* choice is made.

Runs identical eval harness to Phase 7c (same model, same n=200, same seed).
Output: phase7-ablation/results/phase7f_kv_norm_inverted.json
"""
import json
import sys
import time
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME  = "HuggingFaceTB/SmolLM-1.7B"
N_EVAL      = 200
TARGET_P4   = 0.74
FP16_COST   = 1.010
SEED        = 0

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "phase7f_kv_norm_inverted.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor  # noqa: E402


def make_kv_norm_inverted_router(p4=TARGET_P4):
    """TOP-p4 fraction by L2 norm → 4-bit (pathological: big tokens lose precision)."""
    def route(k, li, T):
        norms = k[0].float().norm(dim=-1)           # (T,)
        if T <= 1:
            return [4] * T
        k_top = int(round(p4 * T))
        if k_top >= T:
            return [4] * T
        if k_top == 0:
            return [8] * T
        # smallest value among the top-p4 by norm
        thresh = torch.topk(norms, k_top, largest=True).values.min().item()
        return [4 if norms[t].item() >= thresh else 8 for t in range(T)]
    return route


def eval_hellaswag(model, tokenizer, route_fn, device, n_samples=N_EVAL):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)

    bits_buf = {}

    def make_k_hook(li):
        def hook(module, inp, out):
            T = out.shape[1]
            bits_list = route_fn(out, li, T)
            bits_buf[li] = bits_list
            q = out.clone()
            for t, b in enumerate(bits_list):
                q[0, t] = quantize_tensor(out[0, t].float(), b).to(out.dtype)
            return q
        return hook

    def make_v_hook(li):
        def hook(module, inp, out):
            bits_list = bits_buf.get(li, [4] * out.shape[1])
            q = out.clone()
            for t, b in enumerate(bits_list):
                q[0, t] = quantize_tensor(out[0, t].float(), b).to(out.dtype)
            return q
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(li)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(li)))

    correct, total, all_bits = 0, 0, []
    for item in list(ds)[:n_samples]:
        ctx = item["activity_label"] + ": " + item["ctx"]
        scores = []
        for ending in item["endings"]:
            inputs = tokenizer(ctx + " " + ending, return_tensors="pt",
                               truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            bits_buf.clear()
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            scores.append(-out.loss.item())
        if scores.index(max(scores)) == int(item["label"]):
            correct += 1
        total += 1
        for bl in bits_buf.values():
            all_bits.extend(bl)
        if total % 20 == 0:
            print(f"    eval {total}/{n_samples} acc={correct/total:.3f}", flush=True)

    for h in handles:
        h.remove()

    c  = Counter(all_bits)
    n  = max(len(all_bits), 1)
    p4 = c.get(4, 0) / n; p8 = c.get(8, 0) / n
    cost = p4 * 0.29 + p8 * 0.56
    return {
        "accuracy_pct": round(100.0 * correct / total, 2),
        "avg_bits":     round(4 * p4 + 8 * p8, 3),
        "fpga_cost":    round(cost, 4),
        "fpga_speedup": round(FP16_COST / cost, 3) if cost > 0 else 0,
        "p4_pct":       round(100 * p4, 2),
        "p8_pct":       round(100 * p8, 2),
        "n_tokens":     n,
    }


def main():
    t_all = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    print(f"Device: {device}  |  seed={SEED}  |  p4={TARGET_P4}  "
          f"|  strategy=kv_norm_inverted", flush=True)

    print("\n=== Loading SmolLM-1.7B ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print("\n=== Eval: kv_norm_inverted ===", flush=True)
    t0 = time.time()
    route = make_kv_norm_inverted_router(p4=TARGET_P4)
    metrics = eval_hellaswag(model, tokenizer, route, device, n_samples=N_EVAL)
    metrics["elapsed_s"] = round(time.time() - t0, 1)
    print(f"  kv_norm_inverted: {metrics}", flush=True)

    result = {
        "experiment": "phase7f_kv_norm_inverted",
        "model":      MODEL_NAME,
        "n_samples":  N_EVAL,
        "seed":       SEED,
        "target_p4":  TARGET_P4,
        "strategy":   "kv_norm_inverted (top-p4 by L2 norm → 4-bit)",
        "metrics":    metrics,
        "note":       "Sanity check: the pathological reverse of Phase 7c's kv_norm. "
                      "Tokens with the largest L2 norm — carrying the most attention "
                      "signal — lose precision to 4-bit, while small-norm tokens get "
                      "the 8-bit slots. Expected to drop well below random/controller.",
        "elapsed_s":  round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)
    print(f"kv_norm_inverted: {metrics['accuracy_pct']}% at "
          f"{metrics['fpga_speedup']}× speedup (p4={metrics['p4_pct']}%).", flush=True)


if __name__ == "__main__":
    main()
