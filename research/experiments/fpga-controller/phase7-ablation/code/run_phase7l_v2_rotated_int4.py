"""
Phase 7l v2: QuaRot-style 4-matrix rotation + static INT4 on matched subsets
=============================================================================
Correct 4-matrix rotation (per-head, baked offline into weights). Prior v1
rotated only W_k and W_v → broke attention math → seed 0 hit 28% (near chance).

Correct scheme (QuaRot, Ashkboos et al. NeurIPS 2024):
  For each attention layer, pick two random orthogonal matrices R_K, R_V of
  shape (d_head, d_head), same for all heads within the layer (simplest variant):

    W_q' = R_K.T @ W_q  (per-head block: rotate head output by R_K)
    W_k' = R_K.T @ W_k  (same R_K — Q·K^T invariant under orthogonal R_K)
    W_v' = R_V.T @ W_v  (rotate V output per-head by R_V)
    W_o' = W_o @ R_V    (W_o input side: unrotate per-head so attn·V·W_o unchanged)

  Math: with these four changes together, the attention module output at FP16
  is IDENTICAL to the unrotated model (within fp16 roundoff). The intermediate
  K, V tensors (what we quantize) now live in a more isotropic space →
  INT4 quantization loses less magnitude.

Pre-flight sanity check: compute logits on a single HellaSwag ending under
both unrotated and rotated models (FP16, no quantization). They must match
to ≤ 1e-2 max-abs-diff. If not, abort without running INT4 eval.

Only after the sanity check passes do we run the 5-subset INT4 eval.

Output: phase7-ablation/results/phase7l_v2_rotated_int4_matched.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B"
N_EVAL     = 500
SEEDS      = [0, 1, 2, 3, 4]
ROT_SEED   = 42                       # master rotation seed
FP16_COST  = 1.010
INT4_COST  = 0.290

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "phase7l_v2_rotated_int4_matched.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor  # noqa: E402


# ----------------------------------------------------------------------------
def hadamard(d: int) -> torch.Tensor:
    """Normalized d×d Hadamard matrix. Requires d = power of 2."""
    assert d & (d - 1) == 0, f"Hadamard requires power-of-2 dim, got {d}"
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(d)


def random_orthogonal(d: int, generator: torch.Generator) -> torch.Tensor:
    """R = H · diag(±1): orthogonal, fast-ish, and decorrelates channels."""
    signs = (torch.randint(0, 2, (d,), generator=generator).float() * 2 - 1)
    return hadamard(d) * signs.unsqueeze(0)  # H @ diag(signs); shape (d, d)


def apply_4matrix_rotation(model, rot_seed=ROT_SEED):
    """Per-layer: generate R_K, R_V of shape (d_head, d_head) and apply to
    q_proj, k_proj, v_proj (left-multiply, per head) and o_proj (right-multiply,
    per head). Preserves attention math exactly (up to fp16 roundoff).

    Uses one R_K and one R_V per LAYER (not per head) for simplicity. Different
    R per layer, same R across heads within a layer. QuaRot does this too.
    """
    gen = torch.Generator().manual_seed(rot_seed)
    n_heads = model.config.num_attention_heads
    n_kv    = model.config.num_key_value_heads
    d_model = model.config.hidden_size
    d_head  = d_model // n_heads
    assert n_heads == n_kv, "script assumes full MHA (SmolLM-1.7B has 32/32)"

    for li, layer in enumerate(model.model.layers):
        R_K = random_orthogonal(d_head, gen).to(layer.self_attn.q_proj.weight.device) \
                                             .to(layer.self_attn.q_proj.weight.dtype)
        R_V = random_orthogonal(d_head, gen).to(layer.self_attn.v_proj.weight.device) \
                                             .to(layer.self_attn.v_proj.weight.dtype)

        with torch.no_grad():
            # W_q, W_k, W_v: shape (n_heads * d_head, d_model) = (2048, 2048)
            # Reshape to (n_heads, d_head, d_model), left-multiply by R.T on dim 1
            for proj, R in [(layer.self_attn.q_proj, R_K),
                            (layer.self_attn.k_proj, R_K),
                            (layer.self_attn.v_proj, R_V)]:
                W = proj.weight.data
                W_r = W.view(n_heads, d_head, d_model)
                # R.T @ W_r per head: einsum for clarity
                W_rot = torch.einsum('ij,hjk->hik', R.T, W_r)
                proj.weight.data = W_rot.reshape(n_heads * d_head, d_model).contiguous()

            # W_o: shape (d_model, n_heads * d_head)
            # Reshape to (d_model, n_heads, d_head), right-multiply by R_V on dim 2
            W_o = layer.self_attn.o_proj.weight.data
            W_o_r = W_o.view(d_model, n_heads, d_head)
            W_o_rot = torch.einsum('dhj,jk->dhk', W_o_r, R_V)
            layer.self_attn.o_proj.weight.data = W_o_rot.reshape(d_model,
                                                                  n_heads * d_head).contiguous()

    print(f"  rotated Q/K/V/O in {len(model.model.layers)} layers "
          f"(rotation seed {rot_seed})", flush=True)


# ----------------------------------------------------------------------------
def sanity_check(model_unrot, model_rot, tokenizer, device, sample):
    """Compare log-likelihoods on a single HellaSwag example under both models.
    Returns max |Δ loss| across the 4 endings. Should be near-zero (fp16 roundoff)."""
    ctx = sample["activity_label"] + ": " + sample["ctx"]
    max_diff = 0.0
    for ending in sample["endings"]:
        inputs = tokenizer(ctx + " " + ending, return_tensors="pt",
                           truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            l_unrot = model_unrot(**inputs, labels=inputs["input_ids"]).loss.item()
            l_rot   = model_rot(**inputs, labels=inputs["input_ids"]).loss.item()
        diff = abs(l_unrot - l_rot)
        max_diff = max(max_diff, diff)
    return max_diff


def eval_static_int4(model, tokenizer, subset, device):
    def make_hook():
        def hook(module, inp, out):
            q = out.clone()
            for t in range(out.shape[1]):
                q[0, t] = quantize_tensor(out[0, t].float(), 4).to(out.dtype)
            return q
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook()))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook()))

    correct, total = 0, 0
    for item in subset:
        ctx = item["activity_label"] + ": " + item["ctx"]
        scores = []
        for ending in item["endings"]:
            inputs = tokenizer(ctx + " " + ending, return_tensors="pt",
                               truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            scores.append(-out.loss.item())
        if scores.index(max(scores)) == int(item["label"]):
            correct += 1
        total += 1
        if total % 50 == 0:
            print(f"    eval {total}/{len(subset)} acc={correct/total:.3f}",
                  flush=True)

    for h in handles:
        h.remove()
    return correct / total


# ----------------------------------------------------------------------------
def main():
    t_all = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  n_samples={N_EVAL}  seeds={SEEDS}  rot_seed={ROT_SEED}",
          flush=True)

    print("\n=== Loading SmolLM-1.7B (unrotated reference) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_unrot = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model_unrot.eval()

    print("\n=== Loading second copy + applying 4-matrix rotation ===", flush=True)
    model_rot = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model_rot.eval()
    apply_4matrix_rotation(model_rot, rot_seed=ROT_SEED)

    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    ds_list = list(ds)
    total_N = len(ds_list)

    # --- Sanity check: rotated-FP16 must match unrotated-FP16 ---
    print("\n=== Sanity check: rotated FP16 == unrotated FP16 (single item) ===",
          flush=True)
    max_diff = sanity_check(model_unrot, model_rot, tokenizer, device, ds_list[0])
    print(f"  max |Δ loss| over 4 endings of item 0: {max_diff:.6f}", flush=True)
    if max_diff > 0.01:
        print(f"  ABORT: rotated FP16 diverges from unrotated (max_diff={max_diff:.4f}); "
              "rotation implementation is buggy.", flush=True)
        sys.exit(1)
    print("  PASSED ✓  (≤ 0.01, consistent with fp16 roundoff)", flush=True)

    # --- Free the unrotated model to save VRAM, then run INT4 eval on rotated ---
    del model_unrot
    torch.cuda.empty_cache()

    fp16_path = OUT_DIR / "fp16_multiseed_matched.json"
    fp16_per_seed = {}
    if fp16_path.exists():
        fp16_per_seed = json.load(open(fp16_path))["per_seed"]

    per_seed = {}
    for s in SEEDS:
        print(f"\n=== Seed {s} (matched subset, rotated model, static INT4) ===",
              flush=True)
        rng = np.random.default_rng(seed=s)
        idx = rng.choice(total_N, size=N_EVAL, replace=False)
        subset = [ds_list[int(i)] for i in idx]
        t0 = time.time()
        acc = eval_static_int4(model_rot, tokenizer, subset, device)
        elapsed = time.time() - t0
        fp16_acc = fp16_per_seed.get(str(s), {}).get("accuracy_pct")
        per_seed[str(s)] = {
            "accuracy_pct":      round(100.0 * acc, 2),
            "fpga_speedup":      round(FP16_COST / INT4_COST, 3),
            "fp16_matched":      fp16_acc,
            "delta_vs_fp16_pp":  round(100*acc - fp16_acc, 2) if fp16_acc else None,
            "elapsed_s":         round(elapsed, 1),
        }
        print(f"  seed {s}: {per_seed[str(s)]}", flush=True)

    accs = np.array([per_seed[str(s)]["accuracy_pct"] for s in SEEDS])
    mean = float(accs.mean()); std = float(accs.std(ddof=1))
    deltas = [per_seed[str(s)]["delta_vs_fp16_pp"] for s in SEEDS
              if per_seed[str(s)]["delta_vs_fp16_pp"] is not None]
    delta_mean = float(np.mean(deltas)) if deltas else None
    delta_std  = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None

    summary = {
        "accuracy_mean":               round(mean, 2),
        "accuracy_std":                round(std, 3),
        "accuracy_range":              [round(float(accs.min()), 2),
                                        round(float(accs.max()), 2)],
        "paired_delta_vs_fp16_mean_pp": round(delta_mean, 2) if delta_mean else None,
        "paired_delta_vs_fp16_std_pp":  round(delta_std, 2) if delta_std else None,
        "sanity_check_max_diff":        round(max_diff, 6),
    }

    result = {
        "experiment":    "phase7l_v2_rotated_static_int4_matched",
        "model":         MODEL_NAME,
        "n_samples":     N_EVAL,
        "seeds":         SEEDS,
        "rotation_seed": ROT_SEED,
        "rotation_type": "4-matrix QuaRot-style (Q, K, V, O) baked into weights",
        "per_seed":      per_seed,
        "summary":       summary,
        "compare_unrotated_phase7j": {
            "int4_mean":                  59.08,
            "int4_std":                   2.24,
            "delta_vs_fp16_mean_pp":      -5.76,
            "delta_vs_fp16_std_pp":       0.95,
        },
        "elapsed_s":     round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)
    print(f"\nRotated static INT4 (5 matched subsets, n={N_EVAL}): "
          f"{mean:.2f}% ± {std:.2f}pp", flush=True)
    if delta_mean is not None:
        print(f"Paired Δ vs FP16: mean {delta_mean:+.2f}pp, std ±{delta_std:.2f}pp",
              flush=True)
        print(f"(Unrotated Phase 7j was −5.76 ± 0.95pp; rotation recovers "
              f"{-5.76 - delta_mean:+.2f}pp)", flush=True)


if __name__ == "__main__":
    main()
