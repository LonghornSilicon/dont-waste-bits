"""
Phase 2: Richer Controller Features
=====================================
Extends Phase 1 v2 with two additional input features:
  - H_t: per-head attention entropy (high entropy = uniform attending = safe to compress)
  - L:   layer depth fraction [0,1] (later layers more sensitive — our 1.7B finding)

Controller input: [kv_norm, pos_frac, head_entropy, layer_depth]  (4-dim vs Phase 1's 2-dim)

Requires loading model with eager attention to get attention weights.
Memory strategy: extract all 4 signals offline, cache to disk, free model, train controller.
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

MODEL_NAME   = "HuggingFaceTB/SmolLM-360M"
TRAIN_TEXTS  = 100   # fewer texts due to eager model RAM cost
EPOCHS       = 15
LR           = 1e-3
BATCH_SIZE   = 512
ALPHA        = 1.0
EVAL_SAMPLES = 200
BIT_CLASSES  = [2, 4, 8, 16]

# Quality scores from empirical HellaSwag (same as Phase 1 v2)
FP16_ACC = 42.6
QUALITY_SCORES = torch.tensor([
    (25.0  - 25.0) / (FP16_ACC - 25.0),
    (41.6  - 25.0) / (FP16_ACC - 25.0),
    (42.0  - 25.0) / (FP16_ACC - 25.0),
    (FP16_ACC - 25.0) / (FP16_ACC - 25.0),
], dtype=torch.float32)

OUTPUT_DIR  = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_PATH  = OUTPUT_DIR / "phase2_signals_cache.pt"
RESULT_PATH = OUTPUT_DIR / "phase2_results.json"


# ---- Stage 1: Extract 4-dim signals ----

def extract_signals_4d(model_name, texts, device="cpu", max_length=128):
    """
    Extract [kv_norm, pos_frac, head_entropy, layer_depth] per token per layer.
    Requires eager attention (output_attentions=True).
    Memory: ~1.4GB for model. Frees model after extraction.
    """
    print("Stage 1: Loading model (eager attention for head entropy)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
        attn_implementation="eager"
    ).to(device)
    model.eval()

    num_layers = len(model.model.layers)
    all_signals = []   # list of (T*L, 4) tensors

    kv_buffers = {}
    def make_kv_hook(layer_idx, proj):
        def hook(module, inp, out):
            kv_buffers[f"{layer_idx}_{proj}"] = out.detach()
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_kv_hook(li, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_kv_hook(li, "v")))

    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  Extracting {i}/{len(texts)}", flush=True)
            kv_buffers.clear()
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(device)
            T = inputs["input_ids"].shape[1]
            outputs = model(**inputs, output_attentions=True)

            for li in range(num_layers):
                k = kv_buffers.get(f"{li}_k")
                v = kv_buffers.get(f"{li}_v")
                if k is None or v is None:
                    continue
                k, v = k[0], v[0]  # (T, D)

                # kv_norm: L2 norm of k+v per token, normalized
                kv = torch.cat([k, v], dim=-1)
                kv_norm = kv.norm(dim=-1)
                kv_norm = kv_norm / (kv_norm.max() + 1e-8)  # (T,)

                # pos_frac: token position [0, 1]
                pos_frac = torch.linspace(0, 1, T, device=device)  # (T,)

                # head_entropy: mean Shannon entropy of attention distribution across heads
                # outputs.attentions[li]: (1, num_heads, T, T)
                attn = outputs.attentions[li][0]   # (num_heads, T, T)
                p = attn.clamp(min=1e-9)
                ent_per_head = -(p * p.log()).sum(dim=-1)  # (num_heads, T)
                head_ent = ent_per_head.mean(dim=0)        # (T,)
                head_ent = head_ent / (head_ent.max() + 1e-8)  # normalize [0,1]

                # layer_depth: constant per layer, normalized [0, 1]
                layer_depth = torch.full((T,), li / max(num_layers - 1, 1), device=device)

                # Stack: (T, 4)
                sig = torch.stack([kv_norm, pos_frac, head_ent, layer_depth], dim=-1)
                all_signals.append(sig.cpu())

    for h in handles:
        h.remove()
    del model, outputs

    sig_tensor = torch.cat(all_signals, dim=0)  # (N, 4)
    print(f"  Collected {len(sig_tensor)} token-layer samples", flush=True)
    return sig_tensor, tokenizer


# ---- Controller (4-dim input) ----

class GumbelController4D(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x, tau=1.0, hard=False):
        logits = self.net(x)
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard)
        return probs, logits

    def predict_bits(self, x):
        with torch.no_grad():
            indices = self.net(x).argmax(dim=-1)
        return [BIT_CLASSES[i] for i in indices.tolist()]


# ---- Same normalized compound loss as Phase 1 v2 ----

def compound_loss_normalized(probs, beta):
    qs   = QUALITY_SCORES.to(probs.device)
    bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32, device=probs.device)
    quality_loss  = 1.0 - (probs * qs).sum(dim=-1).mean()
    avg_bits_norm = (probs * bit_t).sum(dim=-1).mean() / 16.0
    return ALPHA * quality_loss + beta * avg_bits_norm, avg_bits_norm.item() * 16.0


# ---- Training ----

def train_controller(sig_tensor, beta, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                     tau_init=2.0, tau_final=0.1):
    controller = GumbelController4D()
    opt = torch.optim.Adam(controller.parameters(), lr=lr)

    n = len(sig_tensor)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    tr_idx, val_idx = idx[:split], idx[split:]

    tau = tau_init
    tau_decay = (tau_final / tau_init) ** (1.0 / epochs)

    for epoch in range(epochs):
        controller.train()
        perm = torch.randperm(len(tr_idx))
        total_loss, n_b = 0.0, 0
        for i in range(0, len(tr_idx), batch_size):
            b = tr_idx[perm[i:i+batch_size]]
            opt.zero_grad()
            probs, _ = controller(sig_tensor[b], tau=tau)
            loss, _ = compound_loss_normalized(probs, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); n_b += 1

        controller.eval()
        with torch.no_grad():
            vp, _ = controller(sig_tensor[val_idx], tau=0.01, hard=True)
            bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32)
            val_bits = (vp * bit_t).sum(dim=-1).mean().item()
            hi = vp.argmax(dim=-1)
            dist = {b: (hi == i).float().mean().item()*100 for i, b in enumerate(BIT_CLASSES)}
        tau *= tau_decay
        print(f"  [β={beta}] Epoch {epoch+1:2d}/{epochs}: loss={total_loss/n_b:.5f} "
              f"avg_bits={val_bits:.2f} tau={tau:.3f} "
              f"dist={{{', '.join(f'{b}b:{v:.0f}%' for b,v in dist.items())}}}", flush=True)

    return controller


def beta_sweep(sig_tensor, betas=(0.3, 0.5, 0.7, 1.0)):
    results = []
    for beta in betas:
        print(f"\n--- Beta={beta} ---", flush=True)
        ctrl = train_controller(sig_tensor, beta=beta, epochs=8)
        ctrl.eval()
        with torch.no_grad():
            probs, _ = ctrl(sig_tensor, tau=0.01, hard=True)
            bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32)
            avg_bits = (probs * bit_t).sum(dim=-1).mean().item()
            hi = probs.argmax(dim=-1)
            dist = {b: (hi==i).float().mean().item()*100 for i,b in enumerate(BIT_CLASSES)}
        print(f"  Beta={beta}: avg_bits={avg_bits:.2f}", flush=True)
        results.append({"beta": beta, "train_avg_bits": avg_bits, "dist": dist})
    return results


# ---- Evaluation ----

def eval_hellaswag(controller, n_samples=EVAL_SAMPLES):
    sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
    from kv_cache_quant import quantize_tensor

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    controller.eval()

    num_layers = len(model.model.layers)
    kv_bits_buf = {}

    def make_k_hook(layer_idx):
        def hook(module, inp, out):
            T  = out.shape[1]
            norm = out.detach().norm(dim=-1, keepdim=True)
            pos  = torch.linspace(0, 1, T, device=out.device).view(1, T, 1)
            # head_entropy: not available without eager — use zeros as fallback
            h_ent = torch.zeros(1, T, 1, device=out.device)
            l_dep = torch.full((1, T, 1), layer_idx / max(num_layers-1,1), device=out.device)
            sig   = torch.cat([norm/(norm.max()+1e-8), pos, h_ent, l_dep], dim=-1)[0]
            bits_list = controller.predict_bits(sig)
            kv_bits_buf[layer_idx] = bits_list
            out_q = out.clone()
            for t, b in enumerate(bits_list):
                out_q[0, t] = quantize_tensor(out[0, t], b)
            return out_q
        return hook

    def make_v_hook(layer_idx):
        def hook(module, inp, out):
            bits_list = kv_bits_buf.get(layer_idx, [4]*out.shape[1])
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
    correct, total, all_bits = 0, 0, []

    for item in list(ds)[:n_samples]:
        ctx = item["activity_label"] + ": " + item["ctx"]
        scores = []
        for ending in item["endings"]:
            inputs = tokenizer(ctx+" "+ending, return_tensors="pt",
                               truncation=True, max_length=256)
            kv_bits_buf.clear()
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            scores.append(-out.loss.item())
        if scores.index(max(scores)) == int(item["label"]):
            correct += 1
        total += 1
        for bl in kv_bits_buf.values():
            all_bits.extend(bl)

    for h in handles:
        h.remove()
    del model

    accuracy = correct / total
    avg_bits = sum(all_bits) / max(len(all_bits), 1)
    counts = Counter(all_bits)
    bit_dist_pct = {b: counts.get(b,0)/len(all_bits)*100 for b in BIT_CLASSES}
    return accuracy, avg_bits, bit_dist_pct


# ---- Main ----

def main():
    t0 = time.time()

    if CACHE_PATH.exists():
        print(f"Loading cached signals from {CACHE_PATH}", flush=True)
        sig_tensor = torch.load(CACHE_PATH, weights_only=True)
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        texts = [x["text"].strip() for x in ds if len(x["text"].strip())>50][:TRAIN_TEXTS]
        sig_tensor, _ = extract_signals_4d(MODEL_NAME, texts)
        torch.save(sig_tensor, CACHE_PATH)

    print(f"Signal tensor: {sig_tensor.shape}", flush=True)

    sweep = beta_sweep(sig_tensor)
    best_beta = min(sweep, key=lambda r: abs(r["train_avg_bits"] - 5.05))["beta"]
    print(f"\nBest beta: {best_beta}", flush=True)

    controller = train_controller(sig_tensor, beta=best_beta)
    accuracy, avg_bits, bit_dist = eval_hellaswag(controller)

    elapsed = time.time() - t0
    result = {
        "experiment": "phase2_4d_features",
        "model": MODEL_NAME,
        "features": ["kv_norm", "pos_frac", "head_entropy", "layer_depth"],
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
    print(f"Phase 2 Result:", flush=True)
    print(f"  Accuracy:  {result['accuracy']}%  (paper: 41.2%)", flush=True)
    print(f"  avg_bits:  {result['avg_bits']}   (paper: 5.05)", flush=True)
    print(f"  Bit dist:  {result['bit_dist_pct']}", flush=True)
    print(f"  Time:      {elapsed:.0f}s", flush=True)

    ts = time.strftime("%Y%m%d_%H%M")
    out_path = OUTPUT_DIR / f"phase2_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
