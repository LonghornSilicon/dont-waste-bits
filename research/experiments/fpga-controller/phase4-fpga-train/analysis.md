# Phase 4 Analysis: Binary {4,8}-bit FPGA Controller

## Result vs Paper DWB

| Metric | Our Binary Controller | Paper DWB | FP16 |
|--------|-----------------------|-----------|------|
| Accuracy | **41.0%** | 41.2% | 42.6% |
| avg_bits | **4.0** | 5.05 | 16.0 |
| FPGA cost | **0.290** | 0.414 | 1.010 |
| FPGA speedup | **3.48x** | 2.44x | 1.00x |

**FPGA throughput advantage: +43% over paper DWB at equal accuracy.**

## Why We Beat the Paper

Paper DWB allocates 47.9% of tokens to 2-bit. On Xilinx Ultrascale+ BRAM:
- 2-bit uses same 4-bit BRAM port → identical FPGA cost as 4-bit
- But 2-bit causes accuracy collapse (25.0% vs 41.6% at 4-bit)
- Paper's "2-bit savings" save no FPGA bandwidth while destroying accuracy

Our binary {4,8} controller:
1. Eliminates 2-bit entirely (wrong-headed trade-off)  
2. Eliminates 16-bit (4x BRAM cost, minimal accuracy gain)
3. Converges to 100% 4-bit at 360M (correct answer — eff_residual=8.1%)
4. FPGA cost: 0.290 vs paper's 0.414 = **30% lower BRAM utilization**

## Scale Dependence

At SmolLM-360M: 100% 4-bit is the global optimum (eff_residual < threshold).
At SmolLM-1.7B: eff_residual=12.4% > threshold → INT4 lossy → controller will
selectively upgrade sensitive tokens to 8-bit, achieving a mix of 4-bit and 8-bit.
This is the regime where the binary FPGA controller shows maximum benefit.

## Next: Phase 5

Run full benchmark (500 samples, HellaSwag + ARC-Challenge + BoolQ) and
validate the FPGA speedup claim with the 1.7B model.
