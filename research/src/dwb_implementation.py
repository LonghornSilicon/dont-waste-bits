"""
Don't Waste Bits! — Independent Re-implementation from arXiv:2604.04722

Original code not yet public (releases at CVPR June 2026).
Implements all equations from the paper.

Architecture (Section 3.6):
  - 4 input signals: [H_t, R_t, V_t, C_t]
  - 3-layer MLP, hidden_dim=128, ReLU activations
  - 4-class output: {2, 4, 8, 16} bit-widths
  - Training loss: alpha*CE + beta*latency + gamma*quality (Eq. 28)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from dataclasses import dataclass

BIT_CLASSES = [2, 4, 8, 16]
CLASS_TO_IDX = {b: i for i, b in enumerate(BIT_CLASSES)}


# ---- Signals (Section 3.5) ----

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """H_t: Shannon entropy of next-token distribution (Eq. 14)."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def compute_rarity(token_id: int, freq_table: dict) -> float:
    """R_t: rarity = 1 - normalized_frequency (Eq. 15)."""
    total = sum(freq_table.values()) or 1
    freq = freq_table.get(token_id, 0)
    return 1.0 - (freq / total)


def compute_attention_variance(attention_weights: torch.Tensor) -> float:
    """V_t: mean variance of attention distribution across heads (Eq. 16).
    attention_weights: (num_heads, seq_len, seq_len)
    """
    return attention_weights.var(dim=-1).mean().item()


def compute_confidence(logits: torch.Tensor) -> float:
    """C_t: max softmax probability."""
    return F.softmax(logits, dim=-1).max().item()


def extract_signals(
    logits: torch.Tensor,
    token_id: int,
    attention_weights: torch.Tensor,
    freq_table: dict,
) -> torch.Tensor:
    """Pack all 4 signals into s_t = [H_t, R_t, V_t, C_t] (Eq. 17)."""
    H = compute_entropy(logits).item()
    R = compute_rarity(token_id, freq_table)
    V = compute_attention_variance(attention_weights)
    C = compute_confidence(logits)
    return torch.tensor([H, R, V, C], dtype=torch.float32)


# ---- Controller (Section 3.6) ----

class DWBController(nn.Module):
    """3-layer MLP controller. Maps s_t ∈ R^4 → logits ∈ R^4."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(4, hidden_dim)    # Eq. 19
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)  # Eq. 20
        self.layer3 = nn.Linear(hidden_dim, 4)    # Eq. 21
        self.bit_widths = BIT_CLASSES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        return self.layer3(h2)

    def predict(self, x: torch.Tensor) -> list:
        with torch.no_grad():
            classes = self.forward(x).argmax(dim=-1)
        return [self.bit_widths[c] for c in classes.tolist()]


# ---- Training objective (Section 3.8) ----

class DWBLoss(nn.Module):
    """L = alpha*CE + beta*latency + gamma*quality (Eq. 28)."""

    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Default latency costs and quality scores per class
        self.register_buffer("latency_costs",
                             torch.tensor([0.5, 1.0, 2.0, 4.0]))  # 2/4/8/16-bit
        self.register_buffer("quality_scores",
                             torch.tensor([0.60, 0.75, 0.90, 1.0]))

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        L_ce = F.cross_entropy(logits, targets)                         # Eq. 25
        L_lat = (probs * self.latency_costs).sum(dim=-1).mean()         # Eq. 26
        L_qual = 1.0 - (probs * self.quality_scores).sum(dim=-1).mean() # Eq. 27
        return self.alpha * L_ce + self.beta * L_lat + self.gamma * L_qual


# ---- Dataset builder ----

def build_training_dataset(model, tokenizer, texts, device="cpu", max_length=128):
    """Extract (signal, target_bitwidth) pairs from texts.

    Uses eager attention to enable output_attentions=True (required for V_t signal).
    transformers 5.x sdpa attention does not support output_attentions.
    """
    import re
    from transformers import AutoModelForCausalLM

    # Reload with eager attention so we can extract attention weights
    model_name = model.config._name_or_path
    print(f"  Reloading {model_name} with eager attention for signal extraction...", flush=True)
    eager_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32,
        attn_implementation="eager"
    ).to(device)
    eager_model.eval()

    all_signals = []
    freq_counter = Counter()

    for text in texts:
        freq_counter.update(tokenizer.encode(text))

    with torch.no_grad():
        for i, text in enumerate(texts):
            if i > 0 and i % 25 == 0:
                print(f"  Signal extraction: {i}/{len(texts)}", flush=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(device)
            outputs = eager_model(**inputs, output_attentions=True)
            logits_seq = outputs.logits[0]
            # outputs.attentions: tuple of (num_heads, seq, seq) per layer
            if outputs.attentions and len(outputs.attentions) > 0:
                attn = outputs.attentions[-1][0]   # last layer, batch 0
            else:
                # Fallback: uniform attention (V_t = 0)
                seq_len = inputs["input_ids"].shape[1]
                attn = torch.ones(1, seq_len, seq_len) / seq_len
            ids = inputs["input_ids"][0]

            for t in range(len(ids)):
                s = extract_signals(logits_seq[t], ids[t].item(), attn, freq_counter)
                all_signals.append(s)

    del eager_model  # free memory

    signals = torch.stack(all_signals)
    # Importance = mean of normalized signals
    mins, maxs = signals.min(0).values, signals.max(0).values
    norm = (signals - mins) / (maxs - mins + 1e-8)
    importance = norm.mean(dim=1)

    q25 = importance.quantile(0.25).item()
    q50 = importance.quantile(0.50).item()
    q75 = importance.quantile(0.75).item()

    targets = []
    for imp in importance.tolist():
        if imp < q25:   targets.append(CLASS_TO_IDX[2])
        elif imp < q50: targets.append(CLASS_TO_IDX[4])
        elif imp < q75: targets.append(CLASS_TO_IDX[8])
        else:           targets.append(CLASS_TO_IDX[16])

    return signals, torch.tensor(targets, dtype=torch.long)


def train_controller(model, tokenizer, train_texts, epochs=5, lr=3e-3,
                     batch_size=256, device="cpu"):
    """Train DWB controller from token signals extracted from train_texts."""
    print(f"Building training dataset from {len(train_texts)} texts...")
    signals, targets = build_training_dataset(model, tokenizer, train_texts, device)
    print(f"  {len(signals)} token samples collected")

    n = len(signals)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    tr_idx, val_idx = idx[:split], idx[split:]

    controller = DWBController().to(device)
    loss_fn = DWBLoss().to(device)
    opt = torch.optim.Adam(controller.parameters(), lr=lr)

    for epoch in range(epochs):
        controller.train()
        perm = torch.randperm(len(tr_idx))
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(tr_idx), batch_size):
            b = tr_idx[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(controller(signals[b].to(device)), targets[b].to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        controller.eval()
        with torch.no_grad():
            preds = controller(signals[val_idx].to(device)).argmax(dim=-1)
            val_acc = (preds == targets[val_idx].to(device)).float().mean().item()
        print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/n_batches:.4f} val_acc={val_acc:.3f}")

    return controller
