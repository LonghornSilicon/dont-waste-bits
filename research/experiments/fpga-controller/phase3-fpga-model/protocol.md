# Phase 3 Protocol: FPGA Memory Latency Model

**Goal:** Build a cycle-accurate FPGA latency proxy that replaces `avg_bits` in the compound loss.

## Why avg_bits is Wrong for FPGA

On CPU: latency ≈ memory bandwidth × total bits (linear in avg_bits).
On FPGA: latency is determined by BRAM port width, LUT cost, and pipeline depth — non-linear.

Key FPGA insight: 3-bit and 4-bit have the same BRAM cost (both fit in a 4-bit port).
5–8-bit fit in an 8-bit port. 9–16-bit require a 16-bit port or two 8-bit ports.
So the effective cost function has **step discontinuities** at port boundaries.

## FPGA Latency Model

Target architecture: Xilinx Ultrascale+ (common for on-device ML, e.g. Versal AI Core)

```python
def fpga_latency_cycles(bits_per_token, seq_len, num_heads, head_dim):
    """
    bits_per_token: Tensor (T,) — assigned bits
    Returns: scalar latency estimate in cycles
    """
    # BRAM port width quantization (step function, differentiable via STE)
    port_width = torch.where(bits_per_token <= 4,  tensor(4),
                 torch.where(bits_per_token <= 8,  tensor(8),
                                                   tensor(16)))
    
    # Memory read cycles: proportional to port_width (wider = more bandwidth needed)
    # Normalized so 16-bit = 1.0 (matches CPU baseline)
    mem_cycles = port_width / 16.0
    
    # LUT cost for dequantization (4-bit: cheap, 8-bit: moderate, 16-bit: trivial)
    lut_cost = {4: 0.05, 8: 0.10, 16: 0.0}  # relative overhead per token
    
    total = (mem_cycles + lut_factor(bits_per_token)).mean()
    return total
```

### Differentiable Version for Training
The step function is approximated with a smooth sigmoid staircase during training,
then snapped to exact port widths during eval.

## Validation
Compare predicted latency ratios against published FPGA KV-cache results in literature.
If within 20% of reported speedups, model is sufficient for controller training signal.

## Files
- `code/fpga_latency_model.py` — standalone model, importable by Phase 4
- `code/validate_model.py` — sanity checks and literature comparison
- `results/latency_model_validation.json`
- `analysis.md`
