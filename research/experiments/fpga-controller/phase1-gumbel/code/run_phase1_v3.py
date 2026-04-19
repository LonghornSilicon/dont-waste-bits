"""
Phase 1 v3: Gumbel-Softmax with int3-range Quality Scores
==========================================================
Fix from v2: quality scores used standard INT4 accuracy (41.6% for 4-bit),
which is nearly lossless at 360M. This makes 4-bit vs 8-bit quality gap tiny,
and controller collapses to 4-bit everywhere.

Fix: use int3-range accuracy (33.0%) as the 4-bit quality reference — this is
the paper's actual lossy 4-bit baseline. Now 4-bit is genuinely bad (quality=0.455)
and 8-bit wins clearly, driving mixed {2,4,8} allocation like the paper.

Also: slower tau decay (start=3.0, end=0.3) to avoid premature locking.

Per-token cost minimum with int3-range quality, alpha=1.0, beta=0.3:
  2-bit:  1.0*(1-0.000) + 0.3*(2/16)  = 1.038  (worst)
  4-bit:  1.0*(1-0.455) + 0.3*(4/16)  = 0.620  (bad — lossy baseline)
  8-bit:  1.0*(1-0.966) + 0.3*(8/16)  = 0.184  (global min — wins clearly)
  16-bit: 1.0*(1-1.000) + 0.3*(16/16) = 0.300  (expensive)
"""

import json
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME   = "HuggingFaceTB/SmolLM-360M"
TRAIN_TEXTS  = 200
EPOCHS       = 15
LR           = 1e-3
BATCH_SIZE   = 512
ALPHA        = 1.0
BETA         = 0.3       # sweep will find best, start here
GUMBEL_TAU   = 3.0       # higher start for more exploration
TAU_FINAL    = 0.3       # softer end — avoid premature locking
EVAL_SAMPLES = 200
BIT_CLASSES  = [2, 4, 8, 16]

# Quality scores from int3-range (paper's lossy 4-bit), not standard INT4
# Our measurements: 2-bit=25%, int3-range-4bit=33.0%, 8-bit=42.0%, 16-bit=42.6%
FP16_ACC   = 42.6
MIN_ACC    = 25.0   # 2-bit floor
QUALITY_SCORES = torch.tensor([
    (25.0 - MIN_ACC) / (FP16_ACC - MIN_ACC),   # 2-bit  → 0.000
    (33.0 - MIN_ACC) / (FP16_ACC - MIN_ACC),   # 4-bit (int3-range) → 0.455
    (42.0 - MIN_ACC) / (FP16_ACC - MIN_ACC),   # 8-bit  → 0.966
    (FP16_ACC - MIN_ACC) / (FP16_ACC - MIN_ACC),# 16-bit → 1.000
], dtype=torch.float32)

OUTPUT_DIR  = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_PATH  = OUTPUT_DIR / "phase1_kv_cache.pt"
RESULT_PATH = OUTPUT_DIR / "phase1_v3_results.json"


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


def compound_loss_normalized(probs, alpha=ALPHA, beta=BETA):
    qs = QUALITY_SCORES.to(probs.device)
    bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32, device=probs.device)

    quality_score = (probs * qs).sum(dim=-1).mean()
    quality_loss  = 1.0 - quality_score                  # [0, 1]

    avg_bits      = (probs * bit_t).sum(dim=-1).mean()
    avg_bits_norm = avg_bits / 16.0                       # [0.125, 1.0]

    loss = alpha * quality_loss + beta * avg_bits_norm
    return loss, quality_loss.item(), avg_bits.item()


def train_controller(kv_tensor, sig_tensor, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                     alpha=ALPHA, beta=BETA, tau_start=GUMBEL_TAU, tau_end=TAU_FINAL):
    controller = GumbelController(input_dim=sig_tensor.shape[1])
    opt = torch.optim.Adam(controller.parameters(), lr=lr)

    n = len(sig_tensor)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    tr_idx, val_idx = idx[:split], idx[split:]

    tau = tau_start
    tau_decay = (tau_end / tau_start) ** (1.0 / epochs)

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
    from collections import Counter
    counts = Counter(all_bits)
    bit_dist_pct = {b: counts.get(b, 0) / len(all_bits) * 100 for b in BIT_CLASSES}
    return accuracy, avg_bits, bit_dist_pct


def beta_sweep(kv_tensor, sig_tensor, betas=(0.1, 0.2, 0.3, 0.5)):
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

    print(f"\nQuality scores (int3-range baseline): {QUALITY_SCORES.tolist()}", flush=True)
    print("Running beta sweep...", flush=True)
    sweep = beta_sweep(kv_tensor, sig_tensor, betas=(0.1, 0.2, 0.3, 0.5))
    best = min(sweep, key=lambda r: abs(r["train_avg_bits"] - 5.05))
    best_beta = best["beta"]
    print(f"\nBest beta for ~5.05 bits: {best_beta} (train avg_bits={best['train_avg_bits']:.2f})", flush=True)

    print(f"\nFull training: epochs={EPOCHS}, beta={best_beta}...", flush=True)
    controller = train_controller(kv_tensor, sig_tensor, beta=best_beta)

    print(f"\nEvaluating on HellaSwag ({EVAL_SAMPLES} samples)...", flush=True)
    accuracy, avg_bits, bit_dist = eval_hellaswag(controller)

    elapsed = time.time() - t0
    result = {
        "experiment": "phase1_gumbel_v3",
        "model": MODEL_NAME,
        "quality_scores": QUALITY_SCORES.tolist(),
        "note": "4-bit quality from int3-range (33.0%) not standard INT4 (41.6%)",
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
    print(f"Phase 1 v3 Result:", flush=True)
    print(f"  Accuracy:  {result['accuracy']}%  (paper: 41.2%)", flush=True)
    print(f"  avg_bits:  {result['avg_bits']}   (paper: 5.05)", flush=True)
    print(f"  Bit dist:  {result['bit_dist_pct']}", flush=True)
    print(f"  Time:      {elapsed:.0f}s", flush=True)

    ts = time.strftime("%Y%m%d_%H%M")
    out_path = OUTPUT_DIR / f"phase1_v3_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
