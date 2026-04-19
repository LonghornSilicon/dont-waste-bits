"""
Phase 1: Gumbel-Softmax Compound Loss Controller
=================================================
Replaces quartile-classification with a differentiable compound loss.

Key changes from DWB v2:
  - No class labels, no cross-entropy on quartile targets
  - Gumbel-softmax → soft bit allocation (differentiable)
  - Loss = alpha * quant_mse_loss + beta * avg_bits
  - quant_mse_loss uses analytical MSE of KV values under predicted bit width
  - Two-stage training: Stage 1 extracts KV data to disk; Stage 2 trains with no model in memory

Memory design: Stage 1 requires ~1.4 GB (model only, single forward pass).
               Stage 2 requires ~200 MB (controller + saved KV tensors only).
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
TRAIN_TEXTS  = 200   # wikitext-2 samples
EPOCHS       = 15
LR           = 1e-3
BATCH_SIZE   = 512
ALPHA        = 1.0   # quant quality weight
BETA         = 0.1   # compression weight (avg_bits penalty)
GUMBEL_TAU   = 1.0   # temperature; anneal toward 0.1 during training
TAU_FINAL    = 0.1
EVAL_SAMPLES = 200   # fast eval; bump to 500 for final
BIT_CLASSES  = [2, 4, 8, 16]

OUTPUT_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_PATH = OUTPUT_DIR / "phase1_kv_cache.pt"
RESULT_PATH = OUTPUT_DIR / "phase1_results.json"

# ---- Stage 1: Extract KV values and signals ----

def extract_kv_and_signals(model_name, texts, device="cpu", max_length=128):
    """Single forward pass per text: collect KV values and per-token norms."""
    print("Stage 1: Loading model for KV extraction...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.eval()

    all_kv_flat  = []  # (T, head_dim * 2) — k and v concatenated per token
    all_signals  = []  # (T, 2) — [kv_norm, position_frac]

    # Hook k_proj and v_proj to capture outputs
    kv_buffers = {}

    def make_hook(layer_idx, proj_name):
        def hook(module, inp, out):
            kv_buffers[f"{layer_idx}_{proj_name}"] = out.detach()  # (B, T, D)
        return hook

    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(
            make_hook(layer_idx, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(
            make_hook(layer_idx, "v")))

    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 25 == 0:
                print(f"  Extracting {i}/{len(texts)}", flush=True)
            kv_buffers.clear()
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(device)
            T = inputs["input_ids"].shape[1]
            model(**inputs)

            # Average KV across all layers for each token position
            k_layers = [kv_buffers[f"{li}_k"][0] for li in range(len(model.model.layers))
                        if f"{li}_k" in kv_buffers]
            v_layers = [kv_buffers[f"{li}_v"][0] for li in range(len(model.model.layers))
                        if f"{li}_v" in kv_buffers]

            if not k_layers:
                continue

            k_mean = torch.stack(k_layers).mean(0)  # (T, D)
            v_mean = torch.stack(v_layers).mean(0)  # (T, D)
            kv_cat = torch.cat([k_mean, v_mean], dim=-1)  # (T, 2D)

            kv_norm = kv_cat.norm(dim=-1, keepdim=True)    # (T, 1)
            pos_frac = torch.linspace(0, 1, T, device=device).unsqueeze(-1)  # (T, 1)
            signals = torch.cat([kv_norm / (kv_norm.max() + 1e-8), pos_frac], dim=-1)  # (T, 2)

            all_kv_flat.append(kv_cat.cpu())
            all_signals.append(signals.cpu())

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    kv_tensor  = torch.cat(all_kv_flat, dim=0)   # (N, 2D)
    sig_tensor = torch.cat(all_signals, dim=0)    # (N, 2)
    print(f"  Collected {len(kv_tensor)} token samples", flush=True)
    return kv_tensor, sig_tensor, tokenizer


# ---- Gumbel-Softmax Controller ----

class GumbelController(nn.Module):
    """Controller with Gumbel-softmax output for differentiable bit allocation."""

    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),   # logits over {2, 4, 8, 16}
        )
        self.bit_tensor = torch.tensor(BIT_CLASSES, dtype=torch.float32)

    def forward(self, x, tau=1.0, hard=False):
        logits = self.net(x)
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard)  # (B, 4)
        return probs, logits

    def predict_bits(self, x):
        """Hard argmax for inference."""
        with torch.no_grad():
            logits = self.net(x)
            indices = logits.argmax(dim=-1)
        return [BIT_CLASSES[i] for i in indices.tolist()]


# ---- Compound Loss ----

def analytical_quant_mse(kv_values, probs):
    """
    Differentiable quality loss: expected quantization MSE of KV values
    under the predicted bit allocation (Gumbel-soft probabilities).

    kv_values: (B, 2D) — actual KV values from Stage 1
    probs:     (B, 4)  — Gumbel-soft class probabilities

    MSE for b-bit symmetric uniform quantizer:
        E[err²] ≈ var(x) / (3 · 4^b)

    Expected MSE under soft probs:
        E[E[err²]] = Σ_b probs[b] * var(x) / (3 * 4^b)
    """
    kv_var = kv_values.var(dim=-1, unbiased=False)  # (B,)
    inv_quant_levels = torch.tensor(
        [1.0 / (3 * 4**b) for b in BIT_CLASSES],
        dtype=torch.float32, device=probs.device
    )  # (4,)
    # Expected MSE per token
    expected_mse = (probs * inv_quant_levels.unsqueeze(0)).sum(dim=-1) * kv_var  # (B,)
    return expected_mse.mean()


def compound_loss(kv_values, probs, alpha=ALPHA, beta=BETA):
    """L = alpha * quant_mse + beta * avg_bits"""
    bit_tensor = torch.tensor(BIT_CLASSES, dtype=torch.float32, device=probs.device)
    avg_bits = (probs * bit_tensor).sum(dim=-1).mean()
    quality   = analytical_quant_mse(kv_values, probs)
    return alpha * quality + beta * avg_bits, quality.item(), avg_bits.item()


# ---- Training ----

def train_controller(kv_tensor, sig_tensor, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    controller = GumbelController(input_dim=sig_tensor.shape[1])
    opt = torch.optim.Adam(controller.parameters(), lr=lr)

    n = len(sig_tensor)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    tr_idx, val_idx = idx[:split], idx[split:]

    tau = GUMBEL_TAU
    tau_decay = (TAU_FINAL / GUMBEL_TAU) ** (1.0 / epochs)

    best_state = None
    best_val_avg_bits = float("inf")
    best_val_bits_with_good_acc = None

    for epoch in range(epochs):
        controller.train()
        perm = torch.randperm(len(tr_idx))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(tr_idx), batch_size):
            b = tr_idx[perm[i:i + batch_size]]
            sigs  = sig_tensor[b]
            kvs   = kv_tensor[b]

            opt.zero_grad()
            probs, _ = controller(sigs, tau=tau)
            loss, qual, abits = compound_loss(kvs, probs)
            loss.backward()
            nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation: hard assignments
        controller.eval()
        with torch.no_grad():
            val_probs, _ = controller(sig_tensor[val_idx], tau=0.01, hard=True)
            bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32)
            val_avg_bits = (val_probs * bit_t).sum(dim=-1).mean().item()
            # Bit distribution
            hard_idx = val_probs.argmax(dim=-1)
            dist = {b: (hard_idx == i).float().mean().item() * 100
                    for i, b in enumerate(BIT_CLASSES)}

        tau *= tau_decay
        print(f"  Epoch {epoch+1:2d}/{epochs}: loss={total_loss/n_batches:.5f} "
              f"avg_bits={val_avg_bits:.2f} tau={tau:.3f} "
              f"dist={{{', '.join(f'{b}b:{v:.0f}%' for b,v in dist.items())}}}", flush=True)

        if val_avg_bits < best_val_avg_bits:
            best_val_avg_bits = val_avg_bits
            best_state = {k: v.clone() for k, v in controller.state_dict().items()}

    if best_state:
        controller.load_state_dict(best_state)
    return controller


# ---- Evaluation ----

def eval_hellaswag(controller, sig_cache_path, n_samples=EVAL_SAMPLES):
    """Run HellaSwag eval using the controller and kv_cache_quant hooks."""
    sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
    from eval_hellaswag import evaluate_hellaswag
    from kv_cache_quant import quantize_tensor

    # We need to run inference with the controller assigning bits dynamically.
    # Use the existing eval infrastructure but patch in our controller.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()

    # Register hooks that use our controller
    controller.eval()
    kv_norms_buf = {}

    def make_k_hook(layer_idx):
        def hook(module, inp, out):
            # Compute signal for this layer's K output
            norm = out.detach().norm(dim=-1, keepdim=True)  # (B, T, 1)
            T = out.shape[1]
            pos = torch.linspace(0, 1, T, device=out.device).unsqueeze(0).unsqueeze(-1)
            kv_norm_max = norm.max() + 1e-8
            sig = torch.cat([norm / kv_norm_max, pos.expand_as(norm)], dim=-1)  # (B, T, 2)
            sig_flat = sig[0]  # (T, 2)
            with torch.no_grad():
                bits_list = controller.predict_bits(sig_flat)
            kv_norms_buf[layer_idx] = bits_list
            # Quantize
            out_q = out.clone()
            for t, b in enumerate(bits_list):
                out_q[0, t] = quantize_tensor(out[0, t], b)
            return out_q
        return hook

    def make_v_hook(layer_idx):
        def hook(module, inp, out):
            bits_list = kv_norms_buf.get(layer_idx, [4] * out.shape[1])
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
    correct, total, total_bits = 0, 0, 0.0
    bits_counts = {b: 0 for b in BIT_CLASSES}

    for item in list(ds)[:n_samples]:
        ctx = item["activity_label"] + ": " + item["ctx"]
        endings = item["endings"]
        label = int(item["label"])
        scores = []
        for ending in endings:
            text = ctx + " " + ending
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            kv_norms_buf.clear()
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            scores.append(-out.loss.item())
        if scores.index(max(scores)) == label:
            correct += 1
        total += 1

        # Collect bit stats from last forward pass
        for li_bits in kv_norms_buf.values():
            for b in li_bits:
                bits_counts[b] = bits_counts.get(b, 0) + 1
                total_bits += b

    for h in handles:
        h.remove()
    del model

    accuracy = correct / total
    avg_bits = total_bits / max(sum(bits_counts.values()), 1)
    total_assigned = sum(bits_counts.values())
    bit_dist_pct = {b: bits_counts[b] / total_assigned * 100 for b in BIT_CLASSES}

    return accuracy, avg_bits, bit_dist_pct


# ---- Main ----

def main():
    t0 = time.time()

    # Stage 1: extract or load cache
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
        print(f"Saved KV cache to {CACHE_PATH}", flush=True)

    # Stage 2: train controller
    print(f"\nStage 2: Training Gumbel-softmax controller ({len(kv_tensor)} tokens)...", flush=True)
    controller = train_controller(kv_tensor, sig_tensor)

    # Stage 3: evaluate
    print(f"\nStage 3: Evaluating on HellaSwag ({EVAL_SAMPLES} samples)...", flush=True)
    accuracy, avg_bits, bit_dist = eval_hellaswag(controller, CACHE_PATH)

    elapsed = time.time() - t0
    result = {
        "experiment": "phase1_gumbel",
        "model": MODEL_NAME,
        "train_tokens": len(kv_tensor),
        "epochs": EPOCHS,
        "alpha": ALPHA,
        "beta": BETA,
        "tau_initial": GUMBEL_TAU,
        "tau_final": TAU_FINAL,
        "eval_samples": EVAL_SAMPLES,
        "accuracy": round(accuracy * 100, 1),
        "avg_bits": round(avg_bits, 3),
        "bit_dist_pct": {str(k): round(v, 1) for k, v in bit_dist.items()},
        "paper_target_accuracy": 41.2,
        "paper_target_avg_bits": 5.05,
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n{'='*50}", flush=True)
    print(f"Phase 1 Result:", flush=True)
    print(f"  Accuracy:  {result['accuracy']}%  (paper: 41.2%)", flush=True)
    print(f"  avg_bits:  {result['avg_bits']}   (paper: 5.05)", flush=True)
    print(f"  Bit dist:  {result['bit_dist_pct']}", flush=True)
    print(f"  Time:      {elapsed:.0f}s", flush=True)

    ts = time.strftime("%Y%m%d_%H%M")
    out_path = OUTPUT_DIR / f"phase1_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    # Also write to fixed name for easy reading
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
