"""
Phase 5: Binary FPGA Controller on SmolLM-1.7B
================================================
Key insight: SmolLM-1.7B has eff_residual=12.4% > losslessness threshold (8-12%).
INT4 is genuinely lossy at 1.7B (FP16: 49.0%, INT4: 41.1%, -7.9pp).
The binary {4,8} controller should learn to upgrade high-sensitivity tokens to 8-bit.

Expected result: mixed {4,8}-bit allocation, accuracy ~45-49%, FPGA cost ~0.30-0.40

Run on GPU (A4000 or better, 16GB VRAM):
  python run_phase5_1b7.py

Estimated time: ~20-30 min on A4000.
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

MODEL_NAME   = "HuggingFaceTB/SmolLM-1.7B"
TRAIN_TEXTS  = 200
EPOCHS       = 15
LR           = 1e-3
BATCH_SIZE   = 512
EVAL_SAMPLES = 200

BIT_CLASSES = [4, 8]
FPGA_COSTS  = torch.tensor([0.29, 0.56], dtype=torch.float32)

# Quality scores for 1.7B (empirically measured):
#   FP16: 49.0%, INT4: 41.1%, INT8: ~48.5% (near FP16), random: 25%
FP16_ACC = 49.0
QUALITY_SCORES = torch.tensor([
    (41.1 - 25.0) / (FP16_ACC - 25.0),   # 4-bit -> 0.671
    (48.5 - 25.0) / (FP16_ACC - 25.0),   # 8-bit -> 0.979 (estimated near-lossless)
], dtype=torch.float32)

OUTPUT_DIR  = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_PATH  = OUTPUT_DIR / "phase5_1b7_kv_cache.pt"
RESULT_PATH = OUTPUT_DIR / "phase5_1b7_results.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor


class BinaryFPGAController(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
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


def fpga_compound_loss(probs, alpha=1.0, beta=0.5):
    qs    = QUALITY_SCORES.to(probs.device)
    costs = FPGA_COSTS.to(probs.device)
    quality_loss = 1.0 - (probs * qs).sum(dim=-1).mean()
    fpga_cost    = (probs * costs).sum(dim=-1).mean()
    fpga_norm    = fpga_cost / 1.01
    loss = alpha * quality_loss + beta * fpga_norm
    avg_bits = (probs * torch.tensor(BIT_CLASSES, dtype=torch.float32, device=probs.device)
                ).sum(dim=-1).mean()
    return loss, quality_loss.item(), avg_bits.item(), fpga_cost.item()


def extract_signals(model_name, n_texts, cache_path, device="cuda"):
    if cache_path.exists():
        print(f"Loading cached signals from {cache_path}", flush=True)
        data = torch.load(cache_path, weights_only=True)
        return data["signals"]

    print(f"Extracting signals from {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    num_layers = len(model.model.layers)

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 50][:n_texts]

    kv_buf = {}
    handles = []
    def make_hook(li, proj):
        def hook(module, inp, out):
            kv_buf[f"{li}_{proj}"] = out.detach().cpu()
        return hook
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(li, "v")))

    all_signals = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 20 == 0:
                print(f"  {i}/{len(texts)}", flush=True)
            kv_buf.clear()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            T = inputs["input_ids"].shape[1]
            model(**inputs)
            for li in range(num_layers):
                k = kv_buf.get(f"{li}_k")
                v = kv_buf.get(f"{li}_v")
                if k is None or v is None:
                    continue
                kv = torch.cat([k[0], v[0]], dim=-1).float()
                kv_norm = kv.norm(dim=-1)
                kv_norm = kv_norm / (kv_norm.max() + 1e-8)
                pos_frac = torch.linspace(0, 1, T)
                sig = torch.stack([kv_norm, pos_frac], dim=-1)
                all_signals.append(sig)

    for h in handles:
        h.remove()
    del model

    sig_tensor = torch.cat(all_signals, dim=0)
    torch.save({"signals": sig_tensor}, cache_path)
    print(f"Saved {len(sig_tensor)} signals to {cache_path}", flush=True)
    return sig_tensor


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
            val_fpga = (vp * FPGA_COSTS).sum(dim=-1).mean().item()
            hi = vp.argmax(dim=-1)
            dist = {b: (hi==i).float().mean().item()*100 for i,b in enumerate(BIT_CLASSES)}
        tau *= tau_decay
        print(f"  [b={beta}] Epoch {epoch+1:2d}/{epochs}: loss={total_loss/n_b:.5f} "
              f"avg_bits={val_bits:.2f} fpga_cost={val_fpga:.3f} tau={tau:.3f} "
              f"dist={{{', '.join(f'{b}b:{v:.0f}%' for b,v in dist.items())}}}", flush=True)

    return controller


def eval_hellaswag(controller, device="cuda", n_samples=EVAL_SAMPLES):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    controller.eval()

    num_layers = len(model.model.layers)
    kv_bits_buf = {}

    def make_k_hook(layer_idx):
        def hook(module, inp, out):
            T = out.shape[1]
            norm = out.detach().float().norm(dim=-1, keepdim=True).cpu()
            pos  = torch.linspace(0, 1, T).view(T, 1)
            sig  = torch.cat([norm[0]/(norm[0].max()+1e-8), pos], dim=-1)
            bits_list = controller.predict_bits(sig)
            kv_bits_buf[layer_idx] = bits_list
            out_q = out.clone()
            for t, b in enumerate(bits_list):
                out_q[0, t] = quantize_tensor(out[0, t].float(), b).to(out.dtype)
            return out_q
        return hook

    def make_v_hook(layer_idx):
        def hook(module, inp, out):
            bits_list = kv_bits_buf.get(layer_idx, [4]*out.shape[1])
            out_q = out.clone()
            for t, b in enumerate(bits_list):
                out_q[0, t] = quantize_tensor(out[0, t].float(), b).to(out.dtype)
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
            inputs = {k: v.to(device) for k, v in inputs.items()}
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
        if total % 20 == 0:
            print(f"  Eval {total}/{n_samples} acc={correct/total:.3f}", flush=True)

    for h in handles:
        h.remove()
    del model

    accuracy  = correct / total
    avg_bits  = sum(all_bits) / max(len(all_bits), 1)
    avg_fpga  = sum(all_costs) / max(len(all_costs), 1)
    speedup   = 1.01 / avg_fpga
    counts    = Counter(all_bits)
    bit_dist  = {b: counts.get(b,0)/len(all_bits)*100 for b in BIT_CLASSES}
    return accuracy, avg_bits, avg_fpga, speedup, bit_dist


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cpu":
        print("WARNING: CPU mode. 1.7B model requires ~3.4GB RAM (FP16). Will be slow.", flush=True)

    t0 = time.time()
    sig_tensor = extract_signals(MODEL_NAME, TRAIN_TEXTS, CACHE_PATH, device=device)
    print(f"Signals: {sig_tensor.shape}", flush=True)

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

    best = min(sweep, key=lambda r: r["fpga_cost"])
    best_beta = best["beta"]
    print(f"\nBest beta: {best_beta} (fpga_cost={best['fpga_cost']:.3f})", flush=True)

    controller = train_controller(sig_tensor, beta=best_beta)
    accuracy, avg_bits, avg_fpga, speedup, bit_dist = eval_hellaswag(
        controller, device=device
    )

    elapsed = time.time() - t0
    result = {
        "experiment": "phase5_1b7_binary_fpga",
        "model": MODEL_NAME,
        "device": device,
        "bit_classes": BIT_CLASSES,
        "quality_scores": QUALITY_SCORES.tolist(),
        "beta_sweep": sweep,
        "best_beta": best_beta,
        "eval_samples": EVAL_SAMPLES,
        "accuracy": round(accuracy * 100, 1),
        "avg_bits": round(avg_bits, 3),
        "avg_fpga_cost": round(avg_fpga, 3),
        "fpga_speedup_vs_fp16": round(speedup, 2),
        "bit_dist_pct": {str(k): round(v, 1) for k, v in bit_dist.items()},
        "baselines": {
            "paper_dwb_360m": {"accuracy": 41.2, "fpga_cost": 0.414, "fpga_speedup": 2.44},
            "fp16_1b7":       {"accuracy": 49.0, "fpga_cost": 1.010},
            "int4_1b7":       {"accuracy": 41.1, "fpga_cost": 0.290, "fpga_speedup": 3.48},
        },
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n{'='*55}", flush=True)
    print(f"Phase 5 (1.7B) Binary FPGA Controller Result:", flush=True)
    print(f"  Accuracy:     {result['accuracy']}%  (FP16: 49.0%, INT4: 41.1%)", flush=True)
    print(f"  avg_bits:     {result['avg_bits']}", flush=True)
    print(f"  FPGA speedup: {result['fpga_speedup_vs_fp16']}x   (paper DWB: 2.44x)", flush=True)
    print(f"  Bit dist:     {result['bit_dist_pct']}", flush=True)
    print(f"  Time:         {elapsed:.0f}s", flush=True)

    ts = time.strftime("%Y%m%d_%H%M")
    out_path = OUTPUT_DIR / f"phase5_1b7_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
