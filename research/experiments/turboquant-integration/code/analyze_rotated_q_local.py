"""
Offline-rotation diagnostic: does pre-rotating K/V projection weights tighten
the per-layer q4_local distribution on SmolLM-1.7B?

Rationale: our Phase 7 per-layer diagnostic found q4_local spread = 0.28 across
24 layers (L1=0.73, L23=0.45), driven by KV L2 norms growing 3× with depth.
Outlier K/V channels in deep layers overflow INT4's max/7 scale and lose up to
55% of token magnitude.

TurboQuant-style / QuaRot-style trick: multiply W_k and W_v by a fixed random
orthogonal R *offline* (baked into weights for inference). K' = X @ (W_k @ R)
is mathematically equivalent to K @ R, but the rotated activation distribution
is closer to isotropic Gaussian → smaller outliers → tighter INT4 quantization
error. Because R is baked in at inference, there's NO runtime Hadamard hardware
cost --- 130nm-friendly.

Procedure:
  1. Load SmolLM-1.7B.
  2. For each attention layer, replace W_k := W_k @ R, W_v := W_v @ R with a
     fixed random orthogonal R (d_kv × d_kv, WHT with random signs).
  3. Run 30 WikiText-2 texts, hook k_proj + v_proj outputs (now rotated).
  4. Compute q_local(rotated_kv, 4) and q_local(rotated_kv, 8) per token, per
     layer.
  5. Compare to the unrotated Phase 7 diagnostic at
     phase7-ablation/results/per_layer_q_local.json.

Decision rule (arbitrary but concrete):
  - q4_mean_spread shrinks ≥40% (from 0.28 to ≤0.17): rotation is worth a
    full 5-seed HellaSwag sweep at p4=0.96 with rotated weights.
  - shrinks 10-40%: marginal; decide after looking at L23 individually.
  - shrinks <10% or widens: rotation does not help; fall back to the existing
    DWB-TurboQuant idea (2-bit tier replacement).

Output: turboquant-integration/results/rotated_q_local.json
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B"
N_TEXTS    = 30
MAX_LEN    = 128
SEED       = 42

OUT_DIR    = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULT_PATH = OUT_DIR / "rotated_q_local.json"
UNROTATED_REFERENCE = (Path(__file__).parents[2]
                       / "fpga-controller/phase7-ablation/results/per_layer_q_local.json")

# Inline the symmetric per-token INT-b quantizer used by Phase 7
# (not yet exported on this branch; keep the script self-contained).
def quantize_tensor(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return x
    levels = 2 ** (bits - 1) - 1
    scale = x.abs().max() / max(levels, 1)
    if scale == 0:
        return x
    return (x / scale).round().clamp(-levels - 1, levels) * scale


# ----------------------------------------------------------------------------
# Walsh-Hadamard transform (power-of-2 dim). Reused from turboquant_impl.
def hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"Hadamard requires power-of-2 dim, got {d}"
    h = x.clone()
    step = 1
    while step < d:
        for i in range(0, d, step * 2):
            a = h[..., i:i + step].clone()
            b = h[..., i + step:i + step * 2].clone()
            h[..., i:i + step] = a + b
            h[..., i + step:i + step * 2] = a - b
        step *= 2
    if normalize:
        h = h / math.sqrt(d)
    return h


def random_orthogonal_applied_to(w: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Right-multiply w by R = H @ diag(signs), where H is normalized Hadamard.

    w has shape (out_dim, in_dim). We want W' = W @ R but R operates on the
    LAST dim of K = X @ W^T, which is W's first dim (out_dim = d_kv). So:
        K = X @ W^T
        K' = K @ R = X @ W^T @ R
        W'^T = W^T @ R  ⇒  W' = R^T @ W
    Since our Hadamard is real & orthogonal and signs is ±1 diagonal,
    R^T = diag(signs) @ H^T = diag(signs) @ H (Hadamard is its own transpose).
    Applied to rows of W: (R^T W)[i, :] = sum_j (R^T)[i, j] W[j, :].
    We compute it as H applied along w's first dim, then multiplied by signs.
    """
    # Left-multiply by diag(signs) @ H:
    #   first apply Hadamard along out_dim (w's dim 0)
    #   then multiply row i by signs[i]
    w_h = hadamard_transform(w.T, normalize=True).T  # apply H along dim 0
    return w_h * signs.unsqueeze(-1)


def rotate_kv_weights_inplace(model, generator):
    """For each attention layer, right-multiply W_k and W_v by the same random
    orthogonal R. We use one R per layer (deterministic given the generator)."""
    num_layers = len(model.model.layers)
    for li, layer in enumerate(model.model.layers):
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        d_out = k_proj.weight.shape[0]
        assert d_out == v_proj.weight.shape[0], "k/v dim mismatch"
        assert d_out & (d_out - 1) == 0, f"d_kv not power-of-2: {d_out}"
        signs = (torch.randint(0, 2, (d_out,), generator=generator).float()
                 * 2 - 1).to(k_proj.weight.device).to(k_proj.weight.dtype)
        with torch.no_grad():
            # Note: linear weight is (out_features, in_features); we want to
            # rotate the out_features axis. Apply H along dim 0, then scale rows.
            new_k = random_orthogonal_applied_to(k_proj.weight.float(), signs.float())
            new_v = random_orthogonal_applied_to(v_proj.weight.float(), signs.float())
            k_proj.weight.copy_(new_k.to(k_proj.weight.dtype))
            v_proj.weight.copy_(new_v.to(v_proj.weight.dtype))
    print(f"  rotated K/V weights in all {num_layers} layers", flush=True)


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  n_texts={N_TEXTS} max_len={MAX_LEN} seed={SEED}",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    num_layers = len(model.model.layers)
    print(f"  model loaded ({num_layers} layers)", flush=True)

    print("\n=== Baking offline random Hadamard rotation into K/V weights ===",
          flush=True)
    gen = torch.Generator().manual_seed(SEED)
    rotate_kv_weights_inplace(model, gen)

    kv_buf, handles = {}, []

    def make_hook(li, which):
        def hook(module, inp, out):
            kv_buf[f"{li}_{which}"] = out.detach().cpu().float()
        return hook

    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(li, "v")))

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 50][:N_TEXTS]

    per_layer_q4 = {li: [] for li in range(num_layers)}
    per_layer_q8 = {li: [] for li in range(num_layers)}
    per_layer_norms = {li: [] for li in range(num_layers)}

    def qlocal(kv, bits):
        kv_q = torch.stack([quantize_tensor(kv[t], bits) for t in range(kv.shape[0])])
        err  = (kv_q - kv).norm(dim=-1)
        nrm  = kv.norm(dim=-1).clamp(min=1e-8)
        return (1.0 - (err / nrm)).clamp(0, 1)

    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 5 == 0:
                print(f"  text {i+1}/{len(texts)}", flush=True)
            kv_buf.clear()
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            inp = {k: v.to(device) for k, v in inp.items()}
            model(**inp)
            for li in range(num_layers):
                k = kv_buf.get(f"{li}_k"); v = kv_buf.get(f"{li}_v")
                if k is None or v is None:
                    continue
                kv = torch.cat([k[0], v[0]], dim=-1)
                per_layer_q4[li].extend(qlocal(kv, 4).tolist())
                per_layer_q8[li].extend(qlocal(kv, 8).tolist())
                per_layer_norms[li].extend(kv.norm(dim=-1).tolist())

    for h in handles:
        h.remove()
    del model

    summary = []
    for li in range(num_layers):
        q4 = torch.tensor(per_layer_q4[li])
        q8 = torch.tensor(per_layer_q8[li])
        norms = torch.tensor(per_layer_norms[li])
        summary.append({
            "layer":      li,
            "n_tokens":   len(q4),
            "q4_mean":    round(q4.mean().item(), 4),
            "q4_std":     round(q4.std().item(), 4),
            "q8_mean":    round(q8.mean().item(), 4),
            "q8_std":     round(q8.std().item(), 4),
            "gap_mean":   round((q8 - q4).mean().item(), 4),
            "norm_mean":  round(norms.mean().item(), 3),
        })

    q4_means = [s["q4_mean"] for s in summary]
    gap_means = [s["gap_mean"] for s in summary]
    q4_spread = max(q4_means) - min(q4_means)
    gap_spread = max(gap_means) - min(gap_means)

    # Load unrotated reference if available.
    comparison = None
    if UNROTATED_REFERENCE.exists():
        ref = json.load(open(UNROTATED_REFERENCE))
        ref_rows = ref["per_layer"]
        ref_q4 = [r["q4_mean"] for r in ref_rows]
        ref_q4_spread = max(ref_q4) - min(ref_q4)
        spread_reduction = 1.0 - (q4_spread / ref_q4_spread)
        comparison = {
            "unrotated_q4_spread": round(ref_q4_spread, 4),
            "rotated_q4_spread":   round(q4_spread, 4),
            "spread_reduction":    round(spread_reduction, 3),
            "unrotated_q4_min":    round(min(ref_q4), 4),
            "rotated_q4_min":      round(min(q4_means), 4),
            "unrotated_L23_q4":    ref_rows[-1]["q4_mean"],
            "rotated_L23_q4":      summary[-1]["q4_mean"],
        }

    if q4_spread <= 0.10:
        verdict = "ROTATION HELPS SUBSTANTIALLY — run full 5-seed HellaSwag at p4=0.96"
    elif q4_spread <= 0.17:
        verdict = "ROTATION HELPS MODERATELY — worth follow-up experiment"
    elif q4_spread <= 0.25:
        verdict = "MARGINAL — mild tightening; decide after inspecting L23"
    else:
        verdict = "NO MEANINGFUL EFFECT — rotation does not solve the 4-bit outlier problem on SmolLM-1.7B"

    result = {
        "experiment":   "rotated_q_local (offline Hadamard + signs baked into K/V proj weights)",
        "model":        MODEL_NAME,
        "n_texts":      N_TEXTS,
        "max_len":      MAX_LEN,
        "seed":         SEED,
        "num_layers":   num_layers,
        "per_layer":    summary,
        "q4_mean_across_layers": {
            "min":    round(min(q4_means), 4),
            "max":    round(max(q4_means), 4),
            "spread": round(q4_spread, 4),
        },
        "gap_spread":     round(gap_spread, 4),
        "comparison_to_unrotated": comparison,
        "verdict":        verdict,
        "elapsed_s":      round(time.time() - t0, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)

    print("\n=== Per-layer q4_local summary (ROTATED weights) ===", flush=True)
    print(f"{'layer':>5}  {'q4_mean':>8}  {'q8_mean':>8}  {'gap':>7}  {'norm':>8}",
          flush=True)
    for s in summary:
        print(f"{s['layer']:>5}  {s['q4_mean']:>8.4f}  {s['q8_mean']:>8.4f}  "
              f"{s['gap_mean']:>7.4f}  {s['norm_mean']:>8.2f}", flush=True)

    print(f"\nq4 spread across layers: {q4_spread:.4f}", flush=True)
    if comparison:
        print(f"Unrotated spread: {comparison['unrotated_q4_spread']:.4f}  →  "
              f"rotated: {q4_spread:.4f}  "
              f"(reduction: {comparison['spread_reduction']*100:+.1f}%)", flush=True)
        print(f"L23 q4:  unrotated {comparison['unrotated_L23_q4']:.4f}  →  "
              f"rotated {comparison['rotated_L23_q4']:.4f}", flush=True)
    print(f"\n>>> {verdict} <<<", flush=True)


if __name__ == "__main__":
    main()
