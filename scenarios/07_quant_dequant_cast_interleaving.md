+++
id = "07_quant_dequant_cast_interleaving"
name = "Quant / Dequant / Cast Interleaving"
benchmark_kind = "quant_dequant_cast_interleaving"
hardware = "Ascend 910B"

# Problem size.
default_N = 4194304

# Number of repeated quant/dequant/cast + compute cycles in the synthetic benchmark.
default_cycles = 16

# Quantization config for the first implementation.
quant_dtype = "int8"
compute_dtype = "float32"
scale_granularity = "per_tensor"

# Fusion scope = number of consecutive quant/dequant/cast + compute cycles fused into one Triton kernel.
fusion_scopes = [1, 2, 4, 8, 16]

# Runtime config.
block = 256
warmup = 20
repeat = 100
trials = 5
check_n = 4096
output_csv = "results/quant_dequant_cast_interleaving.csv"

expected_curve = "execution time decreases first, then plateaus or rebounds when conversion and register pressure dominate"
+++

# Quant / Dequant / Cast Interleaving

## DAG

This motif studies low-precision data conversion around computation.

A typical quantized inference path looks like:

```text
q_x: int8 / fp8
scale, zero_point
-> dequantize / cast to fp16 or fp32
-> compute
-> requantize / cast
-> q_y
```

A simple int8-style formula is:

```text
x_fp = (q_x - zero_point) * scale
z_fp = compute(x_fp)
q_y  = clamp(round(z_fp / out_scale + out_zero_point), -128, 127)
```

For the first-stage benchmark, this can be repeated as several conversion-compute cycles:

```text
q_0
-> dequant/cast
-> compute_0
-> requant/cast
-> q_1
-> dequant/cast
-> compute_1
-> requant/cast
-> q_2
...
-> q_C
```

## LLM relevance

Quant/dequant/cast patterns are common in LLM inference:

```text
INT8 / INT4 / FP8 weight loading
KV cache quantization and dequantization
activation quantization
mixed-precision attention and MLP paths
MoE expert low-precision compute paths
```

These operations often appear around memory-sensitive regions and can strongly affect fusion decisions.

## Fusion scope

`fusion_group_size` means the number of consecutive quant/dequant/cast + compute cycles fused into one Triton kernel.

Examples:

- `G=1`: every conversion-compute-requant cycle is one kernel.
- `G=4`: four consecutive cycles are fused.
- `G=16`: all cycles are fused into one kernel.

The generator should keep the repeated cycles deterministic so that correctness can be checked against a PyTorch reference.

## Expected behavior

Small-scope fusion reduces repeated global-memory reads/writes of quantized intermediate tensors. However, conversion-heavy kernels carry additional temporary values:

```text
q_x
scale / zero_point
dequantized floating value
compute intermediate
requantized output
```

Potential over-fusion causes:

```text
register pressure
cast / conversion instruction pressure
scale / zero-point handling overhead
packing / unpacking overhead
clamp / round instruction overhead
```

Therefore the expected curve may decrease first, then plateau or rebound when conversion overhead and local resource pressure dominate.

## Implementation notes for the generator

The first implementation can use int8 input/output and float32 internal compute for numerical stability. If int8 store support is problematic in the current Triton-Ascend path, the generator may store the quantized result in int32 or float32 as a compatibility fallback, but it must document the fallback in the generated README and keep the DAG semantics clear.

Correctness should be checked against a PyTorch reference. The benchmark should report only execution time.
