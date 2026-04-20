"""
Phase 7c: Routing-strategy ablation on SmolLM-1.7B (HellaSwag n=200)
====================================================================
Compares three per-token 4-bit/8-bit routing strategies at a fixed 74/26 split
(FPGA speedup ≈ 2.81×):

  1. random       — per-token Bernoulli(p=0.74)
  2. controller   — learned binary Gumbel controller (Phase-5 recipe)
  3. kv_norm      — bottom-74% of tokens per-layer by L2 norm → 4-bit

Based on phase5-benchmark/code/run_phase5_1b7.py and beta_calibration_1b7.py.
Outputs: phase7-ablation/results/phase7c_routing_ablation.json

Run: python run_phase7c_routing_ablation.py
Estimated time: ~45–60 min on A4000 (3 strategies × 200 × 4 endings).
"""
import json
import sys
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------
MODEL_NAME    = "HuggingFaceTB/SmolLM-1.7B"
N_EVAL        = 200
N_TRAIN_TEXTS = 200
TRAIN_EPOCHS  = 10
BATCH_SIZE    = 512
TARGET_P4     = 0.74         # yields FPGA cost 0.3602, speedup 2.81×
SEED          = 0

BIT_CLASSES = [4, 8]
FPGA_COSTS  = torch.tensor([0.29, 0.56], dtype=torch.float32)
FP16_COST   = 1.010
FP16_ACC    = 49.0           # paper Table 3

OUT_DIR     = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)
SIGNAL_CACHE = OUT_DIR / "phase7_signal_cache_v2.pt"
RESULT_PATH  = OUT_DIR / "phase7c_routing_ablation.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor  # noqa: E402


# ----------------------------------------------------------------------------
# Controller (matches Phase 5 binary FPGA controller)
class BinaryFPGAController(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, tau=1.0, hard=False):
        return F.gumbel_softmax(self.net(x), tau=tau, hard=hard)

    def predict_bits(self, x):
        with torch.no_grad():
            idx = self.net(x).argmax(dim=-1)
        return [BIT_CLASSES[i] for i in idx.tolist()]


# ----------------------------------------------------------------------------
# Stage 1: extract [kv_norm, pos, q4_local, q8_local] per-token signals
#   — the lossy regime (1.7B) requires per-token quality proxies to get
#     genuine mixed allocation (global scalar qs produce a hard phase transition).
def extract_signals(model, tokenizer, n_texts, device):
    if SIGNAL_CACHE.exists():
        print(f"Loading cached signals from {SIGNAL_CACHE}", flush=True)
        d = torch.load(SIGNAL_CACHE, weights_only=True)
        return d["signals"], d["q_local"]

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 50][:n_texts]
    num_layers = len(model.model.layers)

    kv_buf, handles = {}, []

    def make_hook(li, which):
        def hook(module, inp, out):
            kv_buf[f"{li}_{which}"] = out.detach().cpu().float()
        return hook

    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(li, "v")))

    all_sig, all_q = [], []

    def qlocal(kv, bits):
        kv_q = torch.stack([quantize_tensor(kv[t], bits) for t in range(kv.shape[0])])
        err  = (kv_q - kv).norm(dim=-1)
        nrm  = kv.norm(dim=-1).clamp(min=1e-8)
        return (1.0 - (err / nrm)).clamp(0, 1)

    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 20 == 0:
                print(f"  signals {i}/{len(texts)}", flush=True)
            kv_buf.clear()
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inp = {k: v.to(device) for k, v in inp.items()}
            T = inp["input_ids"].shape[1]
            model(**inp)
            for li in range(num_layers):
                k = kv_buf.get(f"{li}_k"); v = kv_buf.get(f"{li}_v")
                if k is None or v is None:
                    continue
                kv = torch.cat([k[0], v[0]], dim=-1)
                nrm = kv.norm(dim=-1)
                nrm_n = nrm / (nrm.max() + 1e-8)
                pos = torch.linspace(0, 1, T)
                q4 = qlocal(kv, 4)
                q8 = qlocal(kv, 8)
                all_sig.append(torch.stack([nrm_n, pos, q4, q8], dim=-1))
                all_q.append(torch.stack([q4, q8], dim=-1))

    for h in handles:
        h.remove()
    sig = torch.cat(all_sig, dim=0)
    q   = torch.cat(all_q, dim=0)
    torch.save({"signals": sig, "q_local": q}, SIGNAL_CACHE)
    print(f"Saved {len(sig)} signals → {SIGNAL_CACHE}", flush=True)
    return sig, q


# ----------------------------------------------------------------------------
# Stage 2: train a binary controller with a compound loss whose beta is picked
# so that the resulting 4-bit share targets 74% on held-out signals.
def train_controller(signals, q_local, beta, epochs=TRAIN_EPOCHS, lr=1e-3, seed=SEED):
    """Gumbel-softmax binary controller trained with per-token quality proxies."""
    torch.manual_seed(seed)
    ctrl = BinaryFPGAController(input_dim=signals.shape[1])
    opt  = torch.optim.Adam(ctrl.parameters(), lr=lr)
    tau_sched = torch.linspace(2.0, 0.1, epochs)

    N = len(signals)
    for ep in range(epochs):
        tau = tau_sched[ep].item()
        perm = torch.randperm(N)
        for start in range(0, N, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            b_sig = signals[idx]
            b_q   = q_local[idx]
            opt.zero_grad()
            probs = ctrl(b_sig, tau=tau)
            loss = (1.0 - (probs * b_q).sum(-1).mean()) \
                   + beta * (probs * FPGA_COSTS).sum(-1).mean() / FP16_COST
            loss.backward()
            opt.step()
    return ctrl


def controller_p4(ctrl, signals):
    bits = ctrl.predict_bits(signals)
    c = Counter(bits)
    return c.get(4, 0) / len(bits)


def calibrate_beta(signals, q_local, target_p4=TARGET_P4):
    """Binary search over beta until the controller's 4-bit share matches target."""
    lo, hi = 1.0, 2.2
    best = None
    for _ in range(8):
        beta = (lo + hi) / 2.0
        ctrl = train_controller(signals, q_local, beta, epochs=TRAIN_EPOCHS)
        p4 = controller_p4(ctrl, signals)
        print(f"  [calibrate] beta={beta:.3f} → p4={p4*100:.1f}%", flush=True)
        if best is None or abs(p4 - target_p4) < abs(best[1] - target_p4):
            best = (ctrl, p4, beta)
        if p4 < target_p4:
            lo = beta   # need more FPGA pressure → higher beta
        else:
            hi = beta
    return best  # (controller, p4, beta)


# ----------------------------------------------------------------------------
# Evaluation: HellaSwag with a bits-routing callable.
# route_fn(k_tensor, layer_idx) → list[int] of length T (one bit per token)
def eval_hellaswag(model, tokenizer, route_fn, device, n_samples=N_EVAL):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)

    num_layers = len(model.model.layers)
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

    accuracy = correct / total
    counts   = Counter(all_bits)
    n_tok    = max(len(all_bits), 1)
    p4       = counts.get(4, 0) / n_tok
    p8       = counts.get(8, 0) / n_tok
    avg_bits = 4 * p4 + 8 * p8
    fpga     = p4 * 0.29 + p8 * 0.56
    speedup  = FP16_COST / fpga if fpga > 0 else 0
    return {
        "accuracy_pct": round(100 * accuracy, 2),
        "avg_bits":     round(avg_bits, 3),
        "fpga_cost":    round(fpga, 4),
        "fpga_speedup": round(speedup, 3),
        "p4_pct":       round(100 * p4, 2),
        "p8_pct":       round(100 * p8, 2),
        "n_tokens":     n_tok,
    }


# ----------------------------------------------------------------------------
# Routing strategies
def make_random_router(p4=TARGET_P4, seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    def route(k, li, T):
        u = torch.rand(T, generator=gen)
        return [4 if u[t].item() < p4 else 8 for t in range(T)]
    return route


def make_kv_norm_router(p4=TARGET_P4):
    def route(k, li, T):
        # k: (1, T, C) — per-token L2 norm; bottom p4 fraction → 4-bit
        norms = k[0].float().norm(dim=-1)           # (T,)
        if T <= 1:
            return [4] * T
        k_frac = int(round(p4 * T))
        if k_frac >= T:
            return [4] * T
        if k_frac == 0:
            return [8] * T
        thresh = torch.topk(norms, k_frac, largest=False).values.max().item()
        return [4 if norms[t].item() <= thresh else 8 for t in range(T)]
    return route


def make_controller_router(ctrl):
    """Build 4D features [kv_norm, pos, q4_local, q8_local] from K alone (V shares)."""
    ctrl.eval()
    def route(k, li, T):
        kv = k[0].float().cpu()                     # (T, d_k)
        nrm = kv.norm(dim=-1)
        nrm_n = nrm / (nrm.max() + 1e-8)
        pos = torch.linspace(0, 1, T)
        q4 = 1.0 - ((torch.stack([quantize_tensor(kv[t], 4) for t in range(T)])
                     - kv).norm(dim=-1) / nrm.clamp(min=1e-8))
        q8 = 1.0 - ((torch.stack([quantize_tensor(kv[t], 8) for t in range(T)])
                     - kv).norm(dim=-1) / nrm.clamp(min=1e-8))
        sig = torch.stack([nrm_n, pos, q4.clamp(0, 1), q8.clamp(0, 1)], dim=-1)
        return ctrl.predict_bits(sig)
    return route


# ----------------------------------------------------------------------------
def main():
    t_all = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    print(f"Device: {device}  |  seed={SEED}  |  target p4={TARGET_P4}", flush=True)

    print("\n=== Loading SmolLM-1.7B ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # --- Train controller (calibrate beta so p4 ≈ target on training signals)
    print("\n=== Stage 1: signal extraction ===", flush=True)
    signals, q_local = extract_signals(model, tokenizer, N_TRAIN_TEXTS, device)
    print(f"signals shape: {tuple(signals.shape)}  q_local: {tuple(q_local.shape)}",
          flush=True)
    gap = (q_local[:, 1] - q_local[:, 0]).mean().item()
    print(f"q4_mean={q_local[:,0].mean():.4f}  q8_mean={q_local[:,1].mean():.4f}  "
          f"gap_mean={gap:.4f}  beta*_theory={gap/0.267:.3f}", flush=True)

    print("\n=== Stage 2: calibrate controller beta ===", flush=True)
    ctrl, p4_train, beta_star = calibrate_beta(signals, q_local, target_p4=TARGET_P4)
    print(f"chosen beta={beta_star:.3f}  training-set p4={p4_train*100:.1f}%",
          flush=True)

    # --- Routing evals
    strategies = {
        "random":     make_random_router(p4=TARGET_P4, seed=SEED),
        "controller": make_controller_router(ctrl),
        "kv_norm":    make_kv_norm_router(p4=TARGET_P4),
    }

    per_strategy = {}
    for name, route in strategies.items():
        print(f"\n=== Eval: {name} ===", flush=True)
        t0 = time.time()
        metrics = eval_hellaswag(model, tokenizer, route, device, n_samples=N_EVAL)
        metrics["elapsed_s"] = round(time.time() - t0, 1)
        print(f"  {name}: {metrics}", flush=True)
        per_strategy[name] = metrics

    result = {
        "experiment":     "phase7c_routing_ablation",
        "model":          MODEL_NAME,
        "n_samples":      N_EVAL,
        "seed":           SEED,
        "target_p4":      TARGET_P4,
        "beta_controller": round(beta_star, 3),
        "controller_train_p4": round(p4_train, 3),
        "strategies":     per_strategy,
        "baselines": {
            "fp16_paper":  {"accuracy_pct": 49.0,  "fpga_speedup": 1.00},
            "int4_paper":  {"accuracy_pct": 41.1,  "fpga_speedup": 3.48},
            "dwb_paper":   {"accuracy_pct": 48.6,  "fpga_speedup": 2.44},
        },
        "elapsed_s": round(time.time() - t_all, 1),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULT_PATH}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Phase 7c summary:", flush=True)
    for name, m in per_strategy.items():
        print(f"  {name:12s}  acc={m['accuracy_pct']:.2f}%  "
              f"p4={m['p4_pct']:.1f}%  speedup={m['fpga_speedup']:.2f}×",
              flush=True)


if __name__ == "__main__":
    main()
