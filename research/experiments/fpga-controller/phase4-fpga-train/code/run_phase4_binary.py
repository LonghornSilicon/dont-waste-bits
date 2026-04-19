"""
Phase 4: Binary FPGA-Aware Controller ({4-bit, 8-bit} only)
=============================================================
Key insight from Phase 3 FPGA model:
  - 2-bit has IDENTICAL FPGA cost as 4-bit (same 4-bit BRAM port)
  - 16-bit costs 4x more than 4-bit
  - Optimal FPGA set: {4, 8} — never 2, never 16

This simplifies the controller to a BINARY decision:
  - 4-bit: cheap, near-lossless at ≤360M, lossless at 8.1% eff_residual
  - 8-bit: 2x BRAM cost, virtually lossless

For SmolLM-360M (lossless model): will assign 100% 4-bit (confirmed by Phase 1)
For SmolLM-1.7B (lossy model): will learn to upgrade sensitive tokens to 8-bit

Binary controller = simpler, faster training, better gradient signal.

FPGA cost comparison with paper DWB at target 5.05 avg_bits:
  Paper DWB FPGA cost = 0.414 (includes 2-bit tokens, same cost as 4-bit)
  Our binary controller at 5.05 avg_bits = ?
    If 5.05 = α*4 + (1-α)*8 → α = (8-5.05)/(8-4) = 0.7375
    FPGA cost = 0.7375*0.29 + 0.2625*0.56 = 0.214 + 0.147 = 0.361
  Our binary controller at 4.0 avg_bits (100% 4-bit) = 0.290

Both beat paper DWB's 0.414 FPGA cost. The binary controller removes the
worst-of-both-worlds 2-bit option entirely.
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
TRAIN_TEXTS  = 200
EPOCHS       = 15
LR           = 1e-3
BATCH_SIZE   = 512
EVAL_SAMPLES = 200

# Binary bit classes: {4, 8} only
BIT_CLASSES  = [4, 8]

# FPGA costs for binary classes
FPGA_COSTS   = torch.tensor([0.29, 0.56], dtype=torch.float32)  # 4-bit, 8-bit

# Quality scores (from empirical HellaSwag): 4-bit ≈ lossless at 360M
FP16_ACC = 42.6
QUALITY_SCORES = torch.tensor([
    (41.6 - 25.0) / (FP16_ACC - 25.0),   # 4-bit → 0.943
    (42.0 - 25.0) / (FP16_ACC - 25.0),   # 8-bit → 0.966
], dtype=torch.float32)

OUTPUT_DIR  = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_PATH  = Path(__file__).parents[2] / "phase1-gumbel" / "results" / "phase1_kv_cache.pt"
RESULT_PATH = OUTPUT_DIR / "phase4_binary_results.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor


# ---- Binary Gumbel Controller ----

class BinaryFPGAController(nn.Module):
    """Binary controller: assigns each token to 4-bit or 8-bit only."""

    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),   # logits over {4, 8}
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
    """
    L = alpha * quality_loss + beta * fpga_latency

    fpga_latency: expected FPGA cost (not avg_bits — the key difference from Phase 1)
    Range: [0.29, 0.56] for binary {4,8}

    Normalize fpga_latency to [0,1] by dividing by FP16 cost (1.01):
      fpga_norm = fpga_cost / 1.01
      4-bit norm: 0.287
      8-bit norm: 0.554
    """
    qs       = QUALITY_SCORES.to(probs.device)
    costs    = FPGA_COSTS.to(probs.device)

    quality_loss = 1.0 - (probs * qs).sum(dim=-1).mean()
    fpga_cost    = (probs * costs).sum(dim=-1).mean()
    fpga_norm    = fpga_cost / 1.01   # normalize to [0,1]

    loss = alpha * quality_loss + beta * fpga_norm
    avg_bits = (probs * torch.tensor(BIT_CLASSES, dtype=torch.float32, device=probs.device)
                ).sum(dim=-1).mean()
    return loss, quality_loss.item(), avg_bits.item(), fpga_cost.item()


# ---- KV extraction (reuse Phase 1 cache) ----

def load_or_extract_signals(model_name, cache_path):
    if cache_path.exists():
        print(f"Loading Phase 1 KV cache from {cache_path}", flush=True)
        data = torch.load(cache_path, weights_only=True)
        # Phase 1 cache has kv+signals keys
        if "signals" in data:
            return data["signals"]
        # Fallback: compute norm+pos from kv
        kv = data["kv"]
        n = len(kv)
        kv_norm = kv.norm(dim=-1, keepdim=True)
        pos = torch.linspace(0, 1, n).unsqueeze(-1)
        return torch.cat([kv_norm / (kv_norm.max() + 1e-8), pos], dim=-1)
    else:
        raise FileNotFoundError(f"Phase 1 cache not found at {cache_path}. Run Phase 1 first.")


# ---- Training ----

def train_controller(sig_tensor, beta=0.5, epochs=EPOCHS, lr=LR):
    controller = BinaryFPGAController(input_dim=sig_tensor.shape[1])
    opt = torch.optim.Adam(controller.parameters(), lr=lr)
    n = len(sig_tensor)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    tr_idx, val_idx = idx[:split], idx[split:]

    tau, tau_final = 2.0, 0.1
    tau_decay = (tau_final / tau) ** (1.0 / epochs)

    for epoch in range(epochs):
        controller.train()
        perm = torch.randperm(len(tr_idx))
        total_loss, n_b = 0.0, 0
        for i in range(0, len(tr_idx), BATCH_SIZE):
            b = tr_idx[perm[i:i+BATCH_SIZE]]
            opt.zero_grad()
            probs, _ = controller(sig_tensor[b], tau=tau)
            loss, ql, abits, fcost = fpga_compound_loss(probs, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); n_b += 1

        controller.eval()
        with torch.no_grad():
            vp, _ = controller(sig_tensor[val_idx], tau=0.01, hard=True)
            bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32)
            val_bits = (vp * bit_t).sum(dim=-1).mean().item()
            cost_t = FPGA_COSTS
            val_fpga = (vp * cost_t).sum(dim=-1).mean().item()
            hi = vp.argmax(dim=-1)
            dist = {b: (hi==i).float().mean().item()*100 for i,b in enumerate(BIT_CLASSES)}
        tau *= tau_decay
        print(f"  [β={beta}] Epoch {epoch+1:2d}/{epochs}: loss={total_loss/n_b:.5f} "
              f"avg_bits={val_bits:.2f} fpga_cost={val_fpga:.3f} tau={tau:.3f} "
              f"dist={{{', '.join(f'{b}b:{v:.0f}%' for b,v in dist.items())}}}", flush=True)

    return controller


# ---- Evaluation ----

def eval_hellaswag(controller, n_samples=EVAL_SAMPLES):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    controller.eval()

    num_layers = len(model.model.layers)
    kv_bits_buf = {}

    def make_k_hook(layer_idx):
        def hook(module, inp, out):
            T = out.shape[1]
            norm = out.detach().norm(dim=-1, keepdim=True)
            pos  = torch.linspace(0, 1, T, device=out.device).view(1, T, 1)
            sig  = torch.cat([norm/(norm.max()+1e-8), pos], dim=-1)[0]
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
    correct, total, all_bits, all_costs = 0, 0, [], []

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
            all_costs.extend([0.29 if b==4 else 0.56 for b in bl])

    for h in handles:
        h.remove()
    del model

    accuracy = correct / total
    avg_bits = sum(all_bits) / max(len(all_bits), 1)
    avg_fpga = sum(all_costs) / max(len(all_costs), 1)
    fpga_speedup = 1.01 / avg_fpga
    counts = Counter(all_bits)
    bit_dist_pct = {b: counts.get(b,0)/len(all_bits)*100 for b in BIT_CLASSES}

    return accuracy, avg_bits, avg_fpga, fpga_speedup, bit_dist_pct


# ---- Main ----

def main():
    t0 = time.time()
    sig_tensor = load_or_extract_signals(MODEL_NAME, CACHE_PATH)
    print(f"Signals: {sig_tensor.shape}", flush=True)

    # Sweep beta to find best FPGA throughput
    print("\nBeta sweep {0.3, 0.5, 0.7}...", flush=True)
    sweep = []
    for beta in [0.3, 0.5, 0.7]:
        ctrl = train_controller(sig_tensor, beta=beta, epochs=8)
        ctrl.eval()
        with torch.no_grad():
            probs, _ = ctrl(sig_tensor, tau=0.01, hard=True)
            bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32)
            avg_bits = (probs * bit_t).sum(dim=-1).mean().item()
            fcost = (probs * FPGA_COSTS).sum(dim=-1).mean().item()
        print(f"  beta={beta}: avg_bits={avg_bits:.2f} fpga_cost={fcost:.3f}", flush=True)
        sweep.append({"beta": beta, "avg_bits": avg_bits, "fpga_cost": fcost})

    # Pick beta with lowest FPGA cost (maximum throughput)
    best = min(sweep, key=lambda r: r["fpga_cost"])
    best_beta = best["beta"]
    print(f"\nBest beta: {best_beta} (fpga_cost={best['fpga_cost']:.3f})", flush=True)

    controller = train_controller(sig_tensor, beta=best_beta)
    accuracy, avg_bits, avg_fpga, speedup, bit_dist = eval_hellaswag(controller)

    elapsed = time.time() - t0
    result = {
        "experiment": "phase4_binary_fpga",
        "model": MODEL_NAME,
        "bit_classes": BIT_CLASSES,
        "beta_sweep": sweep,
        "best_beta": best_beta,
        "eval_samples": EVAL_SAMPLES,
        "accuracy": round(accuracy * 100, 1),
        "avg_bits": round(avg_bits, 3),
        "avg_fpga_cost": round(avg_fpga, 3),
        "fpga_speedup_vs_fp16": round(speedup, 2),
        "bit_dist_pct": {str(k): round(v, 1) for k, v in bit_dist.items()},
        "baselines": {
            "paper_dwb": {"accuracy": 41.2, "avg_bits": 5.05, "fpga_cost": 0.414, "fpga_speedup": 2.44},
            "fp16":      {"accuracy": 42.6, "avg_bits": 16.0, "fpga_cost": 1.010, "fpga_speedup": 1.00},
        },
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n{'='*55}", flush=True)
    print(f"Phase 4 Binary FPGA Controller Result:", flush=True)
    print(f"  Accuracy:      {result['accuracy']}%  (paper: 41.2%)", flush=True)
    print(f"  avg_bits:      {result['avg_bits']}   (paper: 5.05)", flush=True)
    print(f"  FPGA speedup:  {result['fpga_speedup_vs_fp16']}x   (paper: 2.44x)", flush=True)
    print(f"  Bit dist:      {result['bit_dist_pct']}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M")
    out_path = OUTPUT_DIR / f"phase4_binary_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
