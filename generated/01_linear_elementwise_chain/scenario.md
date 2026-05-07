+++
id = "01_linear_elementwise_chain"
name = "Linear Elementwise Chain"
benchmark_kind = "linear_elementwise_chain"
hardware = "Ascend 910B"

# Problem size.
default_N = 4194304
default_ops = 32

# Fusion scope = number of consecutive elementwise ops fused into one Triton kernel.
fusion_scopes = [1, 2, 4, 8, 16, 32]

# Runtime config.
block = 256
warmup = 20
repeat = 100
trials = 5
check_n = 4096
output_csv = "results/linear_elementwise_chain.csv"

expected_curve = "execution time decreases first, then enters a plateau"
+++

# Linear Elementwise Chain

## DAG

```text
x -> op1 -> op2 -> op3 -> ... -> opN -> y
```

Each op is a simple but non-trivial elementwise transform. The generated Triton code uses a deterministic chain similar to:

```text
y = tanh(a_i * y + b_i) + 0.125 * sigmoid(c_i * y + d_i) + 0.03125 * y
```

## LLM relevance

This motif abstracts residual-path elementwise processing, activation post-processing, score scale/mask paths, and other linear chains around larger LLM kernels.

## Fusion scope

`fusion_group_size` means the number of consecutive elementwise ops fused into one Triton kernel.

Examples:

- `G=1`: each op is one kernel.
- `G=4`: every 4 consecutive ops are fused.
- `G=32`: all 32 ops are fused into one kernel.

## Expected behavior

The expected curve is mostly monotonic decreasing and then plateauing. It may not easily show a strong over-fusion rebound because this chain can reuse a small number of temporary variables.
