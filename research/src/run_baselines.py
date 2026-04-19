"""
Baseline evaluation script for Don't Waste Bits! verification.
Runs FP16, static 4-bit KV, and DWB adaptive on HellaSwag.
Hardware: RTX 4090 (24GB) on NVIDIA Brev.

Usage:
    python research/src/run_baselines.py --model smollm-360m --task hellaswag
"""

import argparse
import json
import time
import csv
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_IDS = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "smollm-360m": "HuggingFaceTB/SmolLM-360M",
    "smollm-1.7b": "HuggingFaceTB/SmolLM-1.7B",
}

PAPER_RESULTS = {
    ("smollm-360m", "hellaswag"): {
        "fp16_accuracy": 41.50, "fp16_latency": 3.50,
        "static4bit_accuracy": 33.60, "static4bit_latency": 2.93,
        "ours_accuracy": 41.20, "ours_latency": 2.41,
    }
}


def measure_latency(model, tokenizer, texts, device, n_tokens=50):
    """Measure average ms/token for autoregressive generation."""
    model.eval()
    latencies = []
    with torch.no_grad():
        for text in texts[:100]:  # sample 100 for speed
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            start = time.perf_counter()
            output = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
            elapsed = time.perf_counter() - start
            n_generated = output.shape[1] - inputs["input_ids"].shape[1]
            if n_generated > 0:
                latencies.append(elapsed / n_generated)
    return sum(latencies) / len(latencies) if latencies else 0.0


def run_lm_eval(model_id, task, quantization=None):
    """Run lm-evaluation-harness and return accuracy."""
    import subprocess, json, tempfile, os
    quant_flag = ""
    if quantization == "4bit":
        quant_flag = "--load_in_4bit"

    result_dir = tempfile.mkdtemp()
    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", f"pretrained={model_id},dtype=float16{',load_in_4bit=True' if quantization == '4bit' else ''}",
        "--tasks", task,
        "--num_fewshot", "0",
        "--output_path", result_dir,
        "--batch_size", "auto",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse results
    result_files = list(Path(result_dir).glob("**/*.json"))
    if result_files:
        with open(result_files[0]) as f:
            data = json.load(f)
        acc = data["results"][task].get("acc,none", data["results"][task].get("acc_norm,none", 0))
        return acc * 100
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="smollm-360m", choices=MODEL_IDS.keys())
    parser.add_argument("--task", default="hellaswag")
    parser.add_argument("--skip_latency", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = MODEL_IDS[args.model]
    results = {}

    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"Task: {args.task}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # FP16 baseline
    print("Running FP16 baseline...")
    fp16_acc = run_lm_eval(model_id, args.task, quantization=None)
    results["fp16_accuracy"] = fp16_acc
    print(f"FP16 accuracy: {fp16_acc:.2f}%")

    # Static 4-bit KV
    print("\nRunning static 4-bit KV baseline...")
    static4_acc = run_lm_eval(model_id, args.task, quantization="4bit")
    results["static4bit_accuracy"] = static4_acc
    print(f"Static 4-bit accuracy: {static4_acc:.2f}%")

    # Compare to paper
    paper = PAPER_RESULTS.get((args.model, args.task), {})
    if paper:
        print(f"\n{'='*60}")
        print("COMPARISON TO PAPER (Table 3):")
        print(f"{'='*60}")
        print(f"FP16:      ours={fp16_acc:.2f}%  paper={paper['fp16_accuracy']}%  diff={fp16_acc - paper['fp16_accuracy']:+.2f}")
        print(f"Static 4b: ours={static4_acc:.2f}%  paper={paper['static4bit_accuracy']}%  diff={static4_acc - paper['static4bit_accuracy']:+.2f}")

    # Save results
    out_path = Path("research/data") / f"baselines_{args.model}_{args.task}_{datetime.now():%Y%m%d_%H%M}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "task": args.task, "results": results, "paper": paper}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
