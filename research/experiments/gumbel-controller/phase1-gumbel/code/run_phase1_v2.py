"""
Phase 1 v2: Gumbel-Softmax with Normalized Quality Loss
=========================================================
Fix from v1: analytical MSE values (~1e-4) are orders of magnitude smaller
than avg_bits (2-16), causing collapse to 2-bit regardless of alpha/beta.

Fix: use normalized quality scores in [0,1] so quality_loss and avg_bits
are on the same scale. Both terms contribute meaningfully to the gradient.

Quality scores (empirically derived from HellaSwag degradation per bit width):
  2-bit:  0.0  (catastrophic — 25% accuracy from our experiments)
  4-bit:  0.75 (near-lossless at ≤360M — 41.6% vs 42.6% FP16)
  8-bit:  0.98 (lossless)
  16-bit: 1.0  (FP16 baseline)

quality_loss = 1 - (probs @ quality_scores).mean()    → [0, 1]
avg_bits_norm = (probs @ [2,4,8,16]).mean() / 16.0    → [0.125, 1.0]

Loss = alpha * quality_loss + beta * avg_bits_norm

With alpha=1.0, beta=0.5, optimal bit per token:
  2-bit total: 1.0*(1-0.0) + 0.5*(2/16) = 1.0 + 0.0625 = 1.0625
  4-bit total: 1.0*(1-0.75)+ 0.5*(4/16) = 0.25 + 0.125  = 0.375  ← minimum
  8-bit total: 1.0*(1-0.98)+ 0.5*(8/16) = 0.02 + 0.25   = 0.27   ← minimum for lower beta
 16-bit total: 1.0*(1-1.0) + 0.5*(16/16)= 0.0  + 0.5    = 0.5

So with beta=0.5: global min at 8-bit; with beta=1.0: pushed toward 4-bit.
Target beta range for 5.05 avg_bits: need mix of 4-bit and 8-bit.
"""

import json
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Config ----
MODEL_NAME   = "HuggingFaceTB/SmolLM-360M"
TRAIN_TEXTS  = 200
EPOCHS       = 15
LR           = 1e-3
BATCH_SIZE   = 512
ALPHA        = 1.0
BETA         = 0.5      # increased: 0.5 on normalized [0,1] scale
GUMBEL_TAU   = 2.0      # start higher for better exploration
TAU_FINAL    = 0.1
EVAL_SAMPLES = 200
BIT_CLASSES  = [2, 4, 8, 16]

# Quality scores: empirically derived from our HellaSwag experiments
# 2-bit=25%, 4-bit=41.6%, 8-bit=42.0%, 16-bit=42.6%  → normalize to [0,1]
FP16_ACC = 42.6
QUALITY_SCORES = torch.tensor([
    (25.0  - 25.0) / (FP16_ACC - 25.0),   # 2-bit  → 0.0
    (41.6  - 25.0) / (FP16_ACC - 25.0),   # 4-bit  → 0.753
    (42.0  - 25.0) / (FP16_ACC - 25.0),   # 8-bit  → 0.964
    (FP16_ACC - 25.0) / (FP16_ACC - 25.0),# 16-bit → 1.0
], dtype=torch.float32)

OUTPUT_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_PATH = OUTPUT_DIR / "phase1_kv_cache.pt"   # reuse from v1 if present
RESULT_PATH = OUTPUT_DIR / "phase1_v2_results.json"


# ---- Stage 1 (same as v1) ----

def extract_kv_and_signals(model_name, texts, device="cpu", max_length=128):
    print("Stage 1: Loading model for KV extraction...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.eval()

    all_kv_flat, all_signals = [], []
    kv_buffers = {}

    def make_hook(layer_idx, proj_name):
        def hook(module, inp, out):
            kv_buffers[f"{layer_idx}_{proj_name}"] = out.detach()
        return hook

    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(layer_idx, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(layer_idx, "v")))

    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 25 == 0:
                print(f"  Extracting {i}/{len(texts)}", flush=True)
            kv_buffers.clear()
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(device)
            T = inputs["input_ids"].shape[1]
            model(**inputs)

            k_layers = [kv_buffers[f"{li}_k"][0] for li in range(len(model.model.layers))
                        if f"{li}_k" in kv_buffers]
            v_layers = [kv_buffers[f"{li}_v"][0] for li in range(len(model.model.layers))
                        if f"{li}_v" in kv_buffers]
            if not k_layers:
                continue

            k_mean = torch.stack(k_layers).mean(0)
            v_mean = torch.stack(v_layers).mean(0)
            kv_cat = torch.cat([k_mean, v_mean], dim=-1)

            kv_norm = kv_cat.norm(dim=-1, keepdim=True)
            pos_frac = torch.linspace(0, 1, T, device=device).unsqueeze(-1)
            signals = torch.cat([kv_norm / (kv_norm.max() + 1e-8), pos_frac], dim=-1)

            all_kv_flat.append(kv_cat.cpu())
            all_signals.append(signals.cpu())

    for h in handles:
        h.remove()
    del model

    kv_tensor  = torch.cat(all_kv_flat, dim=0)
    sig_tensor = torch.cat(all_signals, dim=0)
    print(f"  Collected {len(kv_tensor)} token samples", flush=True)
    return kv_tensor, sig_tensor, tokenizer


# ---- Controller ----

class GumbelController(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x, tau=1.0, hard=False):
        logits = self.net(x)
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard)
        return probs, logits

    def predict_bits(self, x):
        with torch.no_grad():
            logits = self.net(x)
            indices = logits.argmax(dim=-1)
        return [BIT_CLASSES[i] for i in indices.tolist()]


# ---- Normalized Compound Loss ----

def compound_loss_normalized(probs, alpha=ALPHA, beta=BETA):
    """
    L = alpha * quality_loss + beta * avg_bits_normalized

    Both terms in [0, 1] — equal gradient scale, no collapse.
    """
    qs = QUALITY_SCORES.to(probs.device)
    bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32, device=probs.device)

    quality_score = (probs * qs).sum(dim=-1).mean()          # [0, 1]
    quality_loss = 1.0 - quality_score                        # [0, 1]

    avg_bits = (probs * bit_t).sum(dim=-1).mean()             # [2, 16]
    avg_bits_norm = avg_bits / 16.0                           # [0.125, 1.0]

    loss = alpha * quality_loss + beta * avg_bits_norm
    return loss, quality_loss.item(), avg_bits.item()


# ---- Training ----

def train_controller(kv_tensor, sig_tensor, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                     alpha=ALPHA, beta=BETA):
    controller = GumbelController(input_dim=sig_tensor.shape[1])
    opt = torch.optim.Adam(controller.parameters(), lr=lr)

    n = len(sig_tensor)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    tr_idx, val_idx = idx[:split], idx[split:]

    tau = GUMBEL_TAU
    tau_decay = (TAU_FINAL / GUMBEL_TAU) ** (1.0 / epochs)

    for epoch in range(epochs):
        controller.train()
        perm = torch.randperm(len(tr_idx))
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(tr_idx), batch_size):
            b = tr_idx[perm[i:i + batch_size]]
            opt.zero_grad()
            probs, _ = controller(sig_tensor[b], tau=tau)
            loss, ql, abits = compound_loss_normalized(probs, alpha, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        controller.eval()
        with torch.no_grad():
            val_probs, _ = controller(sig_tensor[val_idx], tau=0.01, hard=True)
            bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32)
            val_avg_bits = (val_probs * bit_t).sum(dim=-1).mean().item()
            hard_idx = val_probs.argmax(dim=-1)
            dist = {b: (hard_idx == i).float().mean().item() * 100
                    for i, b in enumerate(BIT_CLASSES)}

        tau *= tau_decay
        print(f"  Epoch {epoch+1:2d}/{epochs}: loss={total_loss/n_batches:.5f} "
              f"avg_bits={val_avg_bits:.2f} tau={tau:.3f} "
              f"dist={{{', '.join(f'{b}b:{v:.0f}%' for b, v in dist.items())}}}", flush=True)

    return controller


# ---- Evaluation ----

def eval_hellaswag(controller, n_samples=EVAL_SAMPLES):
    sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
    from kv_cache_quant import quantize_tensor

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    controller.eval()

    kv_bits_buf = {}

    def make_k_hook(layer_idx):
        def hook(module, inp, out):
            T = out.shape[1]
            norm = out.detach().norm(dim=-1, keepdim=True)
            pos  = torch.linspace(0, 1, T, device=out.device).unsqueeze(-1).unsqueeze(0)
            sig  = torch.cat([norm / (norm.max() + 1e-8), pos.expand_as(norm)], dim=-1)[0]
            bits_list = controller.predict_bits(sig)
            kv_bits_buf[layer_idx] = bits_list
            out_q = out.clone()
            for t, b in enumerate(bits_list):
                out_q[0, t] = quantize_tensor(out[0, t], b)
            return out_q
        return hook

    def make_v_hook(layer_idx):
        def hook(module, inp, out):
            bits_list = kv_bits_buf.get(layer_idx, [4] * out.shape[1])
            out_q = out.clone()
            for t, b in enumerate(bits_list):
                out_q[0, t] = quantize_tensor(out[0, t], b)
            return out_q
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(li)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(li)))

    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    correct, total = 0, 0
    all_bits = []

    for item in list(ds)[:n_samples]:
        ctx = item["activity_label"] + ": " + item["ctx"]
        scores = []
        for ending in item["endings"]:
            text = ctx + " " + ending
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            kv_bits_buf.clear()
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            scores.append(-out.loss.item())
        if scores.index(max(scores)) == int(item["label"]):
            correct += 1
        total += 1
        for bits_list in kv_bits_buf.values():
            all_bits.extend(bits_list)

    for h in handles:
        h.remove()
    del model

    accuracy  = correct / total
    avg_bits  = sum(all_bits) / max(len(all_bits), 1)
    total_assigned = len(all_bits)
    from collections import Counter
    counts = Counter(all_bits)
    bit_dist_pct = {b: counts.get(b, 0) / total_assigned * 100 for b in BIT_CLASSES}
    return accuracy, avg_bits, bit_dist_pct


# ---- Beta sweep to find optimal ----

def beta_sweep(kv_tensor, sig_tensor, betas=(0.3, 0.5, 0.7, 1.0)):
    """Quick training sweep to find which beta gives ~5.05 avg_bits."""
    results = []
    for beta in betas:
        print(f"\n--- Beta={beta} ---", flush=True)
        ctrl = train_controller(kv_tensor, sig_tensor, epochs=8, beta=beta)
        ctrl.eval()
        with torch.no_grad():
            probs, _ = ctrl(sig_tensor, tau=0.01, hard=True)
            bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32)
            avg_bits = (probs * bit_t).sum(dim=-1).mean().item()
            hard_idx = probs.argmax(dim=-1)
            dist = {b: (hard_idx == i).float().mean().item() * 100
                    for i, b in enumerate(BIT_CLASSES)}
        print(f"  Beta={beta}: avg_bits={avg_bits:.2f} dist={dist}", flush=True)
        results.append({"beta": beta, "train_avg_bits": avg_bits, "dist": dist})
    return results


# ---- Main ----

def main():
    t0 = time.time()

    if CACHE_PATH.exists():
        print(f"Loading cached KV data from {CACHE_PATH}", flush=True)
        cache = torch.load(CACHE_PATH, weights_only=True)
        kv_tensor  = cache["kv"]
        sig_tensor = cache["signals"]
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 50][:TRAIN_TEXTS]
        kv_tensor, sig_tensor, _ = extract_kv_and_signals(MODEL_NAME, texts)
        torch.save({"kv": kv_tensor, "signals": sig_tensor}, CACHE_PATH)

    # Quick beta sweep to find optimal compression
    print("\nRunning beta sweep to find target avg_bits ~5.05...", flush=True)
    sweep = beta_sweep(kv_tensor, sig_tensor, betas=(0.3, 0.5, 0.7, 1.0))
    # Pick beta closest to 5.05 target
    best_beta = min(sweep, key=lambda r: abs(r["train_avg_bits"] - 5.05))["beta"]
    print(f"\nBest beta for ~5.05 bits: {best_beta}", flush=True)

    # Full training with best beta
    print(f"\nFull training: epochs={EPOCHS}, beta={best_beta}...", flush=True)
    controller = train_controller(kv_tensor, sig_tensor, beta=best_beta)

    print(f"\nEvaluating on HellaSwag ({EVAL_SAMPLES} samples)...", flush=True)
    accuracy, avg_bits, bit_dist = eval_hellaswag(controller)

    elapsed = time.time() - t0
    result = {
        "experiment": "phase1_gumbel_v2",
        "model": MODEL_NAME,
        "beta_sweep": sweep,
        "best_beta": best_beta,
        "eval_samples": EVAL_SAMPLES,
        "accuracy": round(accuracy * 100, 1),
        "avg_bits": round(avg_bits, 3),
        "bit_dist_pct": {str(k): round(v, 1) for k, v in bit_dist.items()},
        "paper_target_accuracy": 41.2,
        "paper_target_avg_bits": 5.05,
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n{'='*50}", flush=True)
    print(f"Phase 1 v2 Result:", flush=True)
    print(f"  Accuracy:  {result['accuracy']}%  (paper: 41.2%)", flush=True)
    print(f"  avg_bits:  {result['avg_bits']}   (paper: 5.05)", flush=True)
    print(f"  Bit dist:  {result['bit_dist_pct']}", flush=True)
    print(f"  Time:      {elapsed:.0f}s", flush=True)

    ts = time.strftime("%Y%m%d_%H%M")
    out_path = OUTPUT_DIR / f"phase1_v2_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
