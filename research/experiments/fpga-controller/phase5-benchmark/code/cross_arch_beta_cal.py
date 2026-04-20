"""
Cross-architecture beta* validation: does beta* = gap_mean/0.267 hold for OPT-125M?

Hypothesis: The formula beta* = gap_mean/0.267 is derived from FPGA hardware constants
(c_4=0.290, c_8=0.560, C_FP16=1.010) and the average per-token quality gap. Since the
hardware constants are model-independent, the formula should generalize across architectures.
The gap_mean itself depends on model's KV statistics (differs per model) but the formula
for predicting the phase transition should be universal.

Expected: OPT-125M has different KV statistics than SmolLM, so gap_mean will differ.
But beta* = gap_mean/0.267 should still correctly predict the transition point.

OPT-125M uses k_proj/v_proj separate Linear modules (same hook approach as SmolLM).
"""
import json, sys, time
from pathlib import Path
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

MODELS = [
    {"name": "facebook/opt-125m", "slug": "opt125m"},
]
N_TEXTS   = 10
MAX_LEN   = 64
EPOCHS    = 10
LR        = 1e-3
BATCH_SZ  = 256
BIT_CLASSES = [4, 8]
FPGA_COSTS  = torch.tensor([0.29, 0.56], dtype=torch.float32)
BETAS_SWEEP = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]

OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor


class BinaryFPGAController(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
    def forward(self, x, tau=1.0):
        return F.gumbel_softmax(self.net(x), tau=tau, hard=False)
    def predict_bits(self, x):
        with torch.no_grad():
            idx = self.net(x).argmax(dim=-1)
        return [BIT_CLASSES[i] for i in idx.tolist()]


def quantize_rel_error(kv, bits):
    kv_q = quantize_tensor(kv, bits)
    err = (kv_q - kv).norm(dim=-1)
    nrm = kv.norm(dim=-1).clamp(min=1e-8)
    return (err / nrm).clamp(0, 1)


def extract_signals(model_name, n_texts, max_len, cache_path):
    if cache_path.exists():
        d = torch.load(cache_path, weights_only=True)
        print(f"  Loaded cache: {d['signals'].shape[0]} tokens")
        return d["signals"], d["q_local"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"  Loading {model_name}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 50][:n_texts]

    kv_buf = {}
    handles = []
    def make_hook(key):
        def h(mod, inp, out):
            kv_buf[key] = out.detach().float()
        return h

    # OPT uses decoder.layers[i].self_attn.k_proj / v_proj
    # GPT-2 uses h[i].attn.c_attn (fused — skip if not separate)
    # We'll try both OPT-style and LLaMA-style
    layers = None
    style = "unknown"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # LLaMA/SmolLM style
        layers = model.model.layers
        style = "llama"
    elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        # OPT style
        layers = model.model.decoder.layers
        style = "opt"

    if layers is None:
        print("  Cannot find layers — skipping model")
        return None, None

    for li, layer in enumerate(layers):
        sa = layer.self_attn
        if hasattr(sa, "k_proj"):
            handles.append(sa.k_proj.register_forward_hook(make_hook(f"{li}_k")))
            handles.append(sa.v_proj.register_forward_hook(make_hook(f"{li}_v")))
        else:
            print(f"  Layer {li} has no k_proj — model style: {style}")
            break

    all_sig, all_q = [], []
    T_total = 0
    with torch.no_grad():
        for text in texts:
            kv_buf.clear()
            inp = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
            T = inp["input_ids"].shape[1]
            model(**inp)
            for li in range(len(layers)):
                k = kv_buf.get(f"{li}_k")
                v = kv_buf.get(f"{li}_v")
                if k is None or v is None:
                    continue
                k = k[0]; v = v[0]  # (T, D)
                kv = torch.cat([k, v], dim=-1)
                kv_norm = kv.norm(dim=-1)
                kv_norm_n = kv_norm / (kv_norm.max() + 1e-8)
                pos_frac = torch.linspace(0, 1, T)
                q4 = (1.0 - quantize_rel_error(kv, 4)).clamp(0, 1)
                q8 = (1.0 - quantize_rel_error(kv, 8)).clamp(0, 1)
                sig = torch.stack([kv_norm_n, pos_frac, q4, q8], dim=-1)
                q   = torch.stack([q4, q8], dim=-1)
                all_sig.append(sig); all_q.append(q)
            T_total += T

    for h in handles: h.remove()
    del model

    sigs = torch.cat(all_sig, 0)
    qs   = torch.cat(all_q, 0)
    torch.save({"signals": sigs, "q_local": qs}, cache_path)
    print(f"  Extracted {sigs.shape[0]} tokens from {n_texts} texts ({style} style)")
    return sigs, qs


def train_and_eval(signals, q_local, beta, seed=0):
    torch.manual_seed(seed)
    ctrl = BinaryFPGAController()
    opt  = torch.optim.Adam(ctrl.parameters(), lr=LR)
    N = signals.shape[0]
    tau_sched = torch.linspace(2.0, 0.1, EPOCHS)
    for ep in range(EPOCHS):
        tau = tau_sched[ep].item()
        idx = torch.randperm(N)
        for start in range(0, N, BATCH_SZ):
            bs = signals[idx[start:start+BATCH_SZ]]
            bq = q_local[idx[start:start+BATCH_SZ]]
            opt.zero_grad()
            probs = ctrl(bs, tau=tau)
            loss = (1.0 - (probs * bq).sum(-1).mean()) + beta * (probs * FPGA_COSTS).sum(-1).mean() / 1.01
            loss.backward()
            opt.step()
    bits_pred = ctrl.predict_bits(signals)
    c = Counter(bits_pred)
    total = len(bits_pred)
    return round(100.0 * c.get(4, 0) / total, 1)


def main():
    t0 = time.time()
    all_results = {}

    for minfo in MODELS:
        mname, slug = minfo["name"], minfo["slug"]
        print(f"\n=== {mname} ===", flush=True)
        cache = OUT_DIR / f"cross_arch_{slug}_cache.pt"
        sigs, qs = extract_signals(mname, N_TEXTS, MAX_LEN, cache)
        if sigs is None:
            continue

        gap = (qs[:, 1] - qs[:, 0]).numpy()
        gap_mean = float(gap.mean())
        gap_std  = float(gap.std())
        beta_star = gap_mean / 0.267
        print(f"  gap_mean={gap_mean:.4f}  gap_std={gap_std:.4f}  beta*={beta_star:.3f}", flush=True)

        sweep = {}
        for beta in BETAS_SWEEP:
            p4 = train_and_eval(sigs, qs, beta, seed=42)
            fpga_cost = p4/100 * 0.290 + (1 - p4/100) * 0.560
            speedup = round(1.010 / fpga_cost, 3)
            sweep[str(beta)] = {"p4_pct": p4, "speedup": speedup}
            print(f"  beta={beta}: {p4}% 4-bit -> {speedup}x speedup", flush=True)

        # Find measured transition (where p4 jumps above 0%)
        prev_p4 = 0.0
        transition = None
        for beta in BETAS_SWEEP:
            p4 = sweep[str(beta)]["p4_pct"]
            if p4 > 5.0 and prev_p4 < 5.0:
                transition = (BETAS_SWEEP[max(0, BETAS_SWEEP.index(beta)-1)], beta)
            prev_p4 = p4

        theory_error = abs(beta_star - (transition[1] if transition else float("nan"))) if transition else float("nan")
        print(f"  Predicted beta*={beta_star:.3f}  Measured transition={transition}  error={theory_error:.3f}")

        all_results[slug] = {
            "model": mname, "n_tokens": int(sigs.shape[0]),
            "gap_mean": round(gap_mean, 4), "gap_std": round(gap_std, 4),
            "beta_star_theory": round(beta_star, 3),
            "measured_transition": list(transition) if transition else None,
            "theory_error": round(theory_error, 3) if not np.isnan(theory_error) else None,
            "sweep": sweep,
        }

    # Compare with SmolLM results
    smollm_known = {
        "135M": {"gap_mean": 0.3297, "beta_star": 1.234, "measured": [1.2, 1.3]},
        "360M": {"gap_mean": 0.3367, "beta_star": 1.261, "measured": [1.2, 1.4]},
        "1.7B": {"gap_mean": 0.4235, "beta_star": 1.584, "measured": [1.55, 1.57]},
    }
    all_results["smollm_reference"] = smollm_known

    out_path = OUT_DIR / "cross_arch_validation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}  ({time.time()-t0:.0f}s)")
    return all_results


if __name__ == "__main__":
    main()
