"""
Phase 4: Binary FPGA Controller on SmolLM-1.7B (GPU)
======================================================
This is the key experiment: at 1.7B, INT4 is genuinely lossy (eff_residual=12.4%),
so the binary {4,8} controller has real signal to work with.

Expected outcome (based on mechanistic analysis):
  - Controller learns: high-attention-score tokens → 8-bit; others → 4-bit
  - Target: match paper 48.6% accuracy at avg_bits < 7.0 (i.e., better FPGA cost than INT8)
  - Paper DWB 1.7B: 48.6% @ 5.05 avg_bits, FPGA cost = ?
    (paper uses 2-bit which is free on FPGA, so their effective FPGA cost is higher)

Baselines (from paper Table 3, SmolLM-1.7B, HellaSwag):
  FP16:        49.0% @ 16.0 bits
  Static 4-bit: 41.1% @ 4.0 bits   (lossy — 7.9pp degradation)
  Paper DWB:   48.6% @ 5.05 bits

FPGA costs for binary {4, 8}:
  4-bit: 0.290 (Xilinx Ultrascale+ normalized to FP16=1.0)
  8-bit: 0.560

GPU setup: NVIDIA A4000 (16GB VRAM)
  - SmolLM-1.7B in float16: ~3.4GB
  - Plenty of headroom for signal extraction + eval

Usage:
  python run_phase4_1b7_gpu.py [--device cuda] [--train-texts 500] [--eval-samples 500]
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Config ----
MODEL_NAME    = "HuggingFaceTB/SmolLM-1.7B"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_TEXTS   = 500      # more data, better signal extraction
EPOCHS        = 20       # more epochs — harder task than 360M
LR            = 1e-3
BATCH_SIZE    = 1024     # larger batch on GPU
EVAL_SAMPLES  = 500      # full eval for statistical significance (CI ±4.4pp)
BIT_CLASSES   = [4, 8]
FPGA_COSTS    = torch.tensor([0.29, 0.56], dtype=torch.float32)

# Quality scores for 1.7B — INT4 IS lossy here
# From paper Table 3: FP16=49.0%, static 4-bit=41.1%
# Estimate INT8 ~48.5% (near-lossless, eff_residual=12.4% < typical INT8 threshold)
FP16_ACC_1B7  = 49.0
INT4_ACC_1B7  = 41.1   # paper measurement — genuinely lossy
INT8_ACC_1B7  = 48.5   # estimated near-lossless
MIN_ACC       = 25.0   # 2-bit floor
QUALITY_SCORES = torch.tensor([
    (INT4_ACC_1B7 - MIN_ACC) / (FP16_ACC_1B7 - MIN_ACC),  # 4-bit → 0.667
    (INT8_ACC_1B7 - MIN_ACC) / (FP16_ACC_1B7 - MIN_ACC),  # 8-bit → 0.972
], dtype=torch.float32)

OUTPUT_DIR  = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_PATH  = OUTPUT_DIR / "phase4_1b7_kv_cache.pt"
RESULT_PATH = OUTPUT_DIR / "phase4_1b7_results.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor


# ---- Binary Gumbel Controller ----

class BinaryFPGAController(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):  # larger hidden for 1.7B
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, tau=1.0, hard=False):
        logits = self.net(x)
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard)
        return probs, logits

    def predict_bits(self, x):
        with torch.no_grad():
            indices = self.net(x).argmax(dim=-1)
        return [BIT_CLASSES[i] for i in indices.tolist()]


# ---- FPGA-Aware Compound Loss ----

def fpga_compound_loss(probs, alpha=1.0, beta=0.5):
    qs    = QUALITY_SCORES.to(probs.device)
    costs = FPGA_COSTS.to(probs.device)
    bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32, device=probs.device)

    quality_loss = 1.0 - (probs * qs).sum(dim=-1).mean()
    fpga_cost    = (probs * costs).sum(dim=-1).mean()
    fpga_norm    = fpga_cost / 1.01  # normalize to FP16=1.0

    loss     = alpha * quality_loss + beta * fpga_norm
    avg_bits = (probs * bit_t).sum(dim=-1).mean()
    return loss, quality_loss.item(), avg_bits.item(), fpga_cost.item()


# ---- KV + Signal Extraction (GPU) ----

def extract_signals(model_name, texts, device, max_length=128):
    print(f"Stage 1: Extracting KV signals from {model_name} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    all_signals = []
    kv_buffers = {}

    def make_hook(layer_idx, proj):
        def hook(module, inp, out):
            kv_buffers[f"{layer_idx}_{proj}"] = out.detach().cpu()
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(li, "v")))

    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 50 == 0:
                print(f"  {i}/{len(texts)}", flush=True)
            kv_buffers.clear()
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(device)
            T = inputs["input_ids"].shape[1]
            model(**inputs)

            n_layers = len(model.model.layers)
            k_layers = [kv_buffers[f"{li}_k"][0].float()
                        for li in range(n_layers) if f"{li}_k" in kv_buffers]
            v_layers = [kv_buffers[f"{li}_v"][0].float()
                        for li in range(n_layers) if f"{li}_v" in kv_buffers]
            if not k_layers:
                continue

            k_mean = torch.stack(k_layers).mean(0)   # (T, head_dim*heads)
            v_mean = torch.stack(v_layers).mean(0)
            kv_cat = torch.cat([k_mean, v_mean], dim=-1)

            kv_norm = kv_cat.norm(dim=-1, keepdim=True)
            pos_frac = torch.linspace(0, 1, T).unsqueeze(-1)
            signals = torch.cat([kv_norm / (kv_norm.max() + 1e-8), pos_frac], dim=-1)
            all_signals.append(signals)

    for h in handles:
        h.remove()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    sig_tensor = torch.cat(all_signals, dim=0)
    print(f"  Collected {len(sig_tensor)} token samples", flush=True)
    return sig_tensor


# ---- Training (CPU — controller is tiny) ----

def train_controller(sig_tensor, beta=0.5, epochs=EPOCHS, lr=LR):
    controller = BinaryFPGAController(input_dim=sig_tensor.shape[1])
    opt = torch.optim.Adam(controller.parameters(), lr=lr)

    n = len(sig_tensor)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    tr_idx, val_idx = idx[:split], idx[split:]

    tau, tau_final = 3.0, 0.3
    tau_decay = (tau_final / tau) ** (1.0 / epochs)

    for epoch in range(epochs):
        controller.train()
        perm = torch.randperm(len(tr_idx))
        total_loss, n_b = 0.0, 0
        for i in range(0, len(tr_idx), BATCH_SIZE):
            b = tr_idx[perm[i:i + BATCH_SIZE]]
            opt.zero_grad()
            probs, _ = controller(sig_tensor[b], tau=tau)
            loss, ql, abits, fcost = fpga_compound_loss(probs, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_b += 1

        controller.eval()
        with torch.no_grad():
            vp, _ = controller(sig_tensor[val_idx], tau=0.01, hard=True)
            bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32)
            val_bits = (vp * bit_t).sum(dim=-1).mean().item()
            val_fpga = (vp * FPGA_COSTS).sum(dim=-1).mean().item()
            hi = vp.argmax(dim=-1)
            dist = {b: (hi == i).float().mean().item() * 100
                    for i, b in enumerate(BIT_CLASSES)}

        tau *= tau_decay
        print(f"  [β={beta}] Epoch {epoch+1:2d}/{epochs}: loss={total_loss/n_b:.5f} "
              f"avg_bits={val_bits:.2f} fpga={val_fpga:.3f} tau={tau:.3f} "
              f"dist={{{', '.join(f'{b}b:{v:.0f}%' for b,v in dist.items())}}}", flush=True)

    return controller


# ---- Evaluation (GPU) ----

def eval_hellaswag(controller, n_samples, device):
    print(f"\nLoading {MODEL_NAME} for eval on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    controller.eval()

    kv_bits_buf = {}

    def make_k_hook(li):
        def hook(module, inp, out):
            T = out.shape[1]
            norm = out.detach().float().norm(dim=-1, keepdim=True).cpu()
            pos  = torch.linspace(0, 1, T).unsqueeze(-1).unsqueeze(0)
            sig  = torch.cat([norm / (norm.max() + 1e-8), pos.expand_as(norm)], dim=-1)[0]
            bits_list = controller.predict_bits(sig)
            kv_bits_buf[li] = bits_list
            out_q = out.clone()
            # quantize on device
            for t, b in enumerate(bits_list):
                out_q[0, t] = quantize_tensor(out[0, t].cpu().float(), b).to(out.dtype).to(device)
            return out_q
        return hook

    def make_v_hook(li):
        def hook(module, inp, out):
            bits_list = kv_bits_buf.get(li, [4] * out.shape[1])
            out_q = out.clone()
            for t, b in enumerate(bits_list):
                out_q[0, t] = quantize_tensor(out[0, t].cpu().float(), b).to(out.dtype).to(device)
            return out_q
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(li)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(li)))

    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    correct, total, all_bits, all_costs = 0, 0, [], []

    for i, item in enumerate(list(ds)[:n_samples]):
        if i % 50 == 0:
            print(f"  Eval {i}/{n_samples}", flush=True)
        ctx = item["activity_label"] + ": " + item["ctx"]
        scores = []
        for ending in item["endings"]:
            inputs = tokenizer(ctx + " " + ending, return_tensors="pt",
                               truncation=True, max_length=256).to(device)
            kv_bits_buf.clear()
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            scores.append(-out.loss.item())
        if scores.index(max(scores)) == int(item["label"]):
            correct += 1
        total += 1
        for bl in kv_bits_buf.values():
            all_bits.extend(bl)
            all_costs.extend([0.29 if b == 4 else 0.56 for b in bl])

    for h in handles:
        h.remove()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    accuracy = correct / total
    avg_bits = sum(all_bits) / max(len(all_bits), 1)
    avg_fpga = sum(all_costs) / max(len(all_costs), 1)
    fpga_speedup = 1.01 / avg_fpga
    counts = Counter(all_bits)
    bit_dist_pct = {b: counts.get(b, 0) / len(all_bits) * 100 for b in BIT_CLASSES}
    return accuracy, avg_bits, avg_fpga, fpga_speedup, bit_dist_pct


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--train-texts", type=int, default=TRAIN_TEXTS)
    parser.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES)
    parser.add_argument("--skip-cache", action="store_true",
                        help="Re-extract signals even if cache exists")
    args = parser.parse_args()
    device = args.device

    t0 = time.time()
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    # Stage 1: Extract signals
    if CACHE_PATH.exists() and not args.skip_cache:
        print(f"Loading cached signals from {CACHE_PATH}", flush=True)
        sig_tensor = torch.load(CACHE_PATH, weights_only=True)
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 50][:args.train_texts]
        sig_tensor = extract_signals(MODEL_NAME, texts, device)
        torch.save(sig_tensor, CACHE_PATH)
        print(f"Signals cached to {CACHE_PATH}", flush=True)

    print(f"\nSignal tensor: {sig_tensor.shape}", flush=True)
    print(f"Quality scores (4-bit={QUALITY_SCORES[0]:.3f}, 8-bit={QUALITY_SCORES[1]:.3f})", flush=True)
    print(f"Quality gap: {(QUALITY_SCORES[1]-QUALITY_SCORES[0]):.3f} (at 360M this was 0.021 — tiny)", flush=True)

    # Stage 2: Train controller (CPU — tiny MLP, ~10k params)
    print("\nBeta sweep {0.3, 0.5, 0.7}...", flush=True)
    sweep = []
    for beta in [0.3, 0.5, 0.7]:
        ctrl = train_controller(sig_tensor, beta=beta, epochs=10)
        ctrl.eval()
        with torch.no_grad():
            vp, _ = ctrl(sig_tensor, tau=0.01, hard=True)
            bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32)
            avg_bits = (vp * bit_t).sum(dim=-1).mean().item()
            fcost = (vp * FPGA_COSTS).sum(dim=-1).mean().item()
            hi = vp.argmax(dim=-1)
            dist = {b: (hi == i).float().mean().item() * 100 for i, b in enumerate(BIT_CLASSES)}
        print(f"  beta={beta}: avg_bits={avg_bits:.2f} fpga_cost={fcost:.3f} dist={dist}", flush=True)
        sweep.append({"beta": beta, "avg_bits": round(avg_bits, 3),
                      "fpga_cost": round(fcost, 4), "dist": dist})

    # Pick beta closest to target avg_bits = 6.0 (midpoint, gives ~FPGA cost 0.42)
    # Lower is better for FPGA but we want to maintain 1.7B accuracy
    target_bits = 6.0
    best = min(sweep, key=lambda r: abs(r["avg_bits"] - target_bits))
    best_beta = best["beta"]
    print(f"\nBest beta for ~{target_bits} avg_bits: {best_beta} "
          f"(avg_bits={best['avg_bits']:.2f})", flush=True)

    # Full training
    print(f"\nFull training: epochs={EPOCHS}, beta={best_beta}...", flush=True)
    controller = train_controller(sig_tensor, beta=best_beta)

    # Stage 3: Evaluate on HellaSwag
    print(f"\nEvaluating on HellaSwag ({args.eval_samples} samples)...", flush=True)
    accuracy, avg_bits, avg_fpga, speedup, bit_dist = eval_hellaswag(
        controller, args.eval_samples, device
    )

    elapsed = time.time() - t0
    result = {
        "experiment": "phase4_binary_fpga_1b7",
        "model": MODEL_NAME,
        "device": device,
        "bit_classes": BIT_CLASSES,
        "quality_scores": {
            "4bit": round(QUALITY_SCORES[0].item(), 4),
            "8bit": round(QUALITY_SCORES[1].item(), 4),
            "note": "derived from paper Table 3: 4-bit=41.1%, FP16=49.0% on 1.7B"
        },
        "beta_sweep": sweep,
        "best_beta": best_beta,
        "eval_samples": args.eval_samples,
        "accuracy": round(accuracy * 100, 1),
        "avg_bits": round(avg_bits, 3),
        "avg_fpga_cost": round(avg_fpga, 4),
        "fpga_speedup_vs_fp16": round(speedup, 2),
        "bit_dist_pct": {str(k): round(v, 1) for k, v in bit_dist.items()},
        "baselines": {
            "paper_dwb_1b7": {"accuracy": 48.6, "avg_bits": 5.05,
                               "note": "paper Table 3 — CPU latency optimized"},
            "static_4bit_1b7": {"accuracy": 41.1, "avg_bits": 4.0,
                                 "fpga_cost": 0.290, "fpga_speedup": 3.48},
            "fp16_1b7": {"accuracy": 49.0, "avg_bits": 16.0, "fpga_cost": 1.01},
        },
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n{'='*55}", flush=True)
    print(f"Phase 4 Binary FPGA 1.7B Result:", flush=True)
    print(f"  Accuracy:      {result['accuracy']}%", flush=True)
    print(f"  avg_bits:      {result['avg_bits']}", flush=True)
    print(f"  FPGA speedup:  {result['fpga_speedup_vs_fp16']}x vs FP16", flush=True)
    print(f"  Bit dist:      {result['bit_dist_pct']}", flush=True)
    print(f"  Paper DWB:     48.6% @ 5.05 bits", flush=True)
    print(f"  Time:          {elapsed:.0f}s", flush=True)

    ts = time.strftime("%Y%m%d_%H%M")
    out_path = OUTPUT_DIR / f"phase4_1b7_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
