+++
id = "03_glu_swiglu_gating"
name = "GLU / SwiGLU Gating"
benchmark_kind = "glu_swiglu_gating"
hardware = "Ascend 910B"

# Problem size.
# default_N is the number of element positions processed by the gating path.
default_N = 4194304

# default_stages is used to make the gating motif scalable.
# Each stage is a SwiGLU-style micro block:
#   gate_i = a_i * x + b_i
#   value_i = c_i * x + d_i
#   x = silu(gate_i) * value_i + residual_scale_i * x
default_stages = 16

# Fusion scope = number of consecutive SwiGLU-style gating stages fused into one Triton kernel.
fusion_scopes = [1, 2, 4, 8, 16]

# Runtime config.
block = 256
warmup = 20
repeat = 100
trials = 5
check_n = 4096
output_csv = "results/glu_swiglu_gating.csv"

expected_curve = "execution time decreases first, then plateaus, and may slightly rebound when gating stages become resource-heavy"
+++

# GLU / SwiGLU Gating

## DAG

Real LLM MLP blocks often use a gated FFN structure:

```text
              gate = x @ W_gate
            /
x ---------
            \
              value = x @ W_up

hidden = silu(gate) * value
out    = hidden @ W_down
```

In many implementations, `gate` and `value` are produced by one packed projection and then split:

```text
x
-> packed_linear
-> [gate, value]
-> split
-> silu(gate)
-> silu(gate) * value
-> y
```

For the first-stage microbenchmark, the GEMM itself does not need to be included. The generated benchmark can take `x` as input and apply a deterministic sequence of SwiGLU-style elementwise gating stages:

```text
x_0 = x

for i in 0..S-1:
    gate_i  = a_i * x_i + b_i
    value_i = c_i * x_i + d_i
    x_{i+1} = silu(gate_i) * value_i + r_i * x_i

y = x_S
```

where:

```text
silu(t) = t * sigmoid(t)
```

This keeps the benchmark simple while preserving the key gate/value split-and-merge behavior.

## LLM relevance

This motif abstracts the post-projection gating part of GLU/SwiGLU MLP blocks, which is common in modern LLM FFN layers. It is also representative of small two-path elementwise fan-out/fan-in patterns around larger projection kernels.

## Fusion scope

`fusion_group_size` means the number of consecutive SwiGLU-style gating stages fused into one Triton kernel.

Examples:

- `G=1`: each gating stage is one kernel.
- `G=4`: every 4 consecutive gating stages are fused.
- `G=16`: all 16 gating stages are fused into one kernel.

The generator should preserve this meaning even if it implements the benchmark with separate intermediate buffers for the unfused cases.

## Expected behavior

Small-scope fusion should reduce kernel launches and intermediate global-memory traffic. However, each gating stage introduces multiple temporary values:

```text
gate, sigmoid(gate), silu(gate), value, residual term
```

Therefore, larger fusion scopes can increase register pressure and instruction footprint. The expected curve is usually decreasing first and then plateauing; a slight rebound may appear under stress settings such as larger `default_stages` or more expensive gate/value math.

## Implementation notes for the generator

The first implementation can avoid GEMM and operate on one-dimensional tensors. Correctness should be checked against a PyTorch reference using the same deterministic coefficients.

Suggested implementation:

```text
G < default_stages:
    group_kernel computes G consecutive gating stages and writes an intermediate tensor.
    multiple group kernels are launched sequentially.

G = default_stages:
    one full_fusion_kernel computes all gating stages and writes y.
```

The benchmark should report only execution time.
