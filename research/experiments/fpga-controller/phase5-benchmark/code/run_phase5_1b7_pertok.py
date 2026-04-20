"""
Phase 5 (v2): Binary FPGA Controller on SmolLM-1.7B — Per-Token Quality Signals
==================================================================================
KEY FIX over v1: Uses per-token quantization error as quality proxy instead of
global empirical quality scores.

Problem with global quality scores at 1.7B:
  Global: q4=0.671, q8=0.979 → ALL beta values push to 100% 8-bit (L_8bit < L_4bit)
  → FPGA speedup = 1.010/0.560 = 1.80x — worse than DWB (2.44x)!

Why: Global scores give uniform gradient → no incentive to differentiate tokens.
Controller maximizes quality/cost ratio globally → one class wins everywhere.

Fix: Per-token quality proxy computed during Stage 1:
  q4_local = 1 - ||quantize(kv, 4) - kv|| / ||kv||
  q8_local = 1 - ||quantize(kv, 8) - kv|| / ||kv||

High-kv_norm tokens (larger values, higher INT4 error) get lower q4_local → pushed to 8-bit
Low-kv_norm tokens (smaller values, lower INT4 error) get higher q4_local → stay at 4-bit
→ Genuine mixed {4,8}-bit allocation!

Run on GPU (A4000 or better, 16GB VRAM):
  python run_phase5_1b7_pertok.py

Expected: ~60-80% 4-bit, ~20-40% 8-bit, accuracy ~45-48%, FPGA cost ~0.35-0.42
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

OUTPUT_DIR  = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_PATH  = OUTPUT_DIR / "phase5_1b7_pertok_cache.pt"
RESULT_PATH = OUTPUT_DIR / "phase5_1b7_pertok_results.json"

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from kv_cache_quant import quantize_tensor


class BinaryFPGAControllerPertok(nn.Module):
    """
    Per-token quality-aware binary controller.
    Input: [kv_norm, pos_frac, q4_local, q8_local]  (4D)
    Output: soft probabilities over {4-bit, 8-bit}
    """
    def __init__(self, input_dim=4, hidden_dim=64):
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


def fpga_pertok_loss(probs, q_local, alpha=1.0, beta=0.5):
    """
    Per-token quality-aware compound loss.

    q_local: (N, 2) per-token quality scores for [4-bit, 8-bit]
             q_local[t, b] = 1 - ||quantize(kv_t, b) - kv_t|| / ||kv_t||
             Higher = less error = better quality at that bit-width.

    L = alpha * (1 - E_t[sum_b p_tb * q_local_tb])
      + beta  * (E_t[sum_b p_tb * c_b] / 1.01)
    """
    costs = FPGA_COSTS.to(probs.device)
    q     = q_local.to(probs.device)

    quality_loss = 1.0 - (probs * q).sum(dim=-1).mean()
    fpga_cost    = (probs * costs).sum(dim=-1).mean()
    fpga_norm    = fpga_cost / 1.01

    loss = alpha * quality_loss + beta * fpga_norm
    avg_bits = (probs * torch.tensor(BIT_CLASSES, dtype=torch.float32, device=probs.device)
                ).sum(dim=-1).mean()
    return loss, quality_loss.item(), avg_bits.item(), fpga_cost.item()


def quantize_rel_error(kv: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token relative quantization error (0=perfect, 1=all error)."""
    kv_q = quantize_tensor(kv, bits)
    err = (kv_q - kv).norm(dim=-1)
    nrm = kv.norm(dim=-1).clamp(min=1e-8)
    return (err / nrm).clamp(0, 1)


def extract_signals_pertok(model_name, n_texts, cache_path, device="cuda"):
    if cache_path.exists():
        print(f"Loading cached signals from {cache_path}", flush=True)
        data = torch.load(cache_path, weights_only=True)
        return data["signals"], data["q_local"]

    print(f"Extracting per-token signals from {model_name}...", flush=True)
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
            kv_buf[f"{li}_{proj}"] = out.detach().cpu().float()
        return hook
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(li, "v")))

    all_signals, all_q_local = [], []
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
                k, v = k[0], v[0]  # (T, D)

                # Structural features
                kv = torch.cat([k, v], dim=-1)
                kv_norm = kv.norm(dim=-1)
                kv_norm_n = kv_norm / (kv_norm.max() + 1e-8)
                pos_frac = torch.linspace(0, 1, T)

                # Per-token quality proxies (1 = lossless, 0 = all error)
                err4 = quantize_rel_error(kv, 4)   # (T,)
                err8 = quantize_rel_error(kv, 8)   # (T,)
                q4_local = (1.0 - err4).clamp(0, 1)
                q8_local = (1.0 - err8).clamp(0, 1)

                sig = torch.stack([kv_norm_n, pos_frac, q4_local, q8_local], dim=-1)  # (T, 4)
                q   = torch.stack([q4_local, q8_local], dim=-1)                        # (T, 2)

                all_signals.append(sig)
                all_q_local.append(q)

    for h in handles:
        h.remove()
    del model

    sig_tensor = torch.cat(all_signals, dim=0)  # (N, 4)
    q_tensor   = torch.cat(all_q_local,  dim=0)  # (N, 2)
    torch.save({"signals": sig_tensor, "q_local": q_tensor}, cache_path)
    print(f"Saved {len(sig_tensor)} token signals to {cache_path}", flush=True)

    # Diagnostic: what fraction of tokens have high 4-bit error at 1.7B?
    err4_mean = (1.0 - q_tensor[:, 0]).mean().item()
    err8_mean = (1.0 - q_tensor[:, 1]).mean().item()
    print(f"  Mean 4-bit rel_error: {err4_mean:.4f}  Mean 8-bit rel_error: {err8_mean:.4f}", flush=True)

    return sig_tensor, q_tensor


def train_controller(sig_tensor, q_tensor, beta=0.5, epochs=EPOCHS, lr=LR):
    controller = BinaryFPGAControllerPertok(input_dim=sig_tensor.shape[1])
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
            loss, ql, abits, fcost = fpga_pertok_loss(probs, q_tensor[b], beta=beta)
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
            kv_f = out.detach().float().cpu()  # (1, T, D)
            kv_flat = kv_f[0]                  # (T, D)
            kv_norm = kv_flat.norm(dim=-1)
            kv_norm_n = kv_norm / (kv_norm.max() + 1e-8)
            pos = torch.linspace(0, 1, T)
            q4 = (1.0 - quantize_rel_error(kv_flat, 4)).clamp(0, 1)
            q8 = (1.0 - quantize_rel_error(kv_flat, 8)).clamp(0, 1)
            sig = torch.stack([kv_norm_n, pos, q4, q8], dim=-1)  # (T, 4)
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

    t0 = time.time()
    sig_tensor, q_tensor = extract_signals_pertok(MODEL_NAME, TRAIN_TEXTS, CACHE_PATH, device)
    print(f"Signals: {sig_tensor.shape}  Q_local: {q_tensor.shape}", flush=True)

    # Show per-token quality distribution to understand the regime
    q4_mean = q_tensor[:, 0].mean().item()
    q8_mean = q_tensor[:, 1].mean().item()
    print(f"Mean q4_local={q4_mean:.4f}  q8_local={q8_mean:.4f}", flush=True)
    print(f"Tokens with q4 < 0.85 (high error): {(q_tensor[:,0] < 0.85).float().mean().item()*100:.1f}%", flush=True)

    print("\nBeta sweep {0.3, 0.5, 0.7}...", flush=True)
    sweep = []
    for beta in [0.3, 0.5, 0.7]:
        ctrl = train_controller(sig_tensor, q_tensor, beta=beta, epochs=8)
        ctrl.eval()
        with torch.no_grad():
            probs, _ = ctrl(sig_tensor, tau=0.01, hard=True)
            bit_t = torch.tensor(BIT_CLASSES, dtype=torch.float32)
            avg_bits = (probs * bit_t).sum(dim=-1).mean().item()
            fcost = (probs * FPGA_COSTS).sum(dim=-1).mean().item()
            hi = probs.argmax(dim=-1)
            dist = {b: (hi==i).float().mean().item()*100 for i,b in enumerate(BIT_CLASSES)}
        print(f"  beta={beta}: avg_bits={avg_bits:.2f} fpga_cost={fcost:.3f} dist={dist}", flush=True)
        sweep.append({"beta": beta, "avg_bits": round(avg_bits,3),
                      "fpga_cost": round(fcost,3), "bit_dist": dist})

    # Pick best beta: lowest FPGA cost (or closest to target if all similar)
    best = min(sweep, key=lambda r: r["fpga_cost"])
    best_beta = best["beta"]
    print(f"\nBest beta: {best_beta} (fpga_cost={best['fpga_cost']:.3f})", flush=True)

    controller = train_controller(sig_tensor, q_tensor, beta=best_beta)
    accuracy, avg_bits, avg_fpga, speedup, bit_dist = eval_hellaswag(controller, device)

    elapsed = time.time() - t0
    result = {
        "experiment": "phase5_1b7_pertok_fpga",
        "method": "per-token quality proxy (q_local = 1 - quant_err/kv_norm)",
        "model": MODEL_NAME,
        "device": device,
        "bit_classes": BIT_CLASSES,
        "q4_mean_train": round(q4_mean, 4),
        "q8_mean_train": round(q8_mean, 4),
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

    print(f"\n{'='*60}", flush=True)
    print(f"Phase 5 (1.7B, per-token) Binary FPGA Controller Result:", flush=True)
    print(f"  Accuracy:     {result['accuracy']}%  (FP16: 49.0%, INT4: 41.1%)", flush=True)
    print(f"  avg_bits:     {result['avg_bits']}", flush=True)
    print(f"  FPGA speedup: {result['fpga_speedup_vs_fp16']}x   (paper DWB: 2.44x, INT4: 3.48x)", flush=True)
    print(f"  Bit dist:     {result['bit_dist_pct']}", flush=True)
    print(f"  Time:         {elapsed:.0f}s", flush=True)

    ts = time.strftime("%Y%m%d_%H%M")
    out_path = OUTPUT_DIR / f"phase5_1b7_pertok_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
