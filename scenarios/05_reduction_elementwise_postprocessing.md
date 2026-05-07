+++
id = "05_reduction_elementwise_postprocessing"
name = "Reduction + Elementwise Post-processing"
benchmark_kind = "reduction_elementwise_postprocessing"
hardware = "Ascend 910B"

# Problem size.
# The default tensor is interpreted as [default_rows, default_cols].
# Reduction is performed along default_cols.
default_rows = 16384
default_cols = 256

# Use an RMSNorm-like reduction + post-processing graph for the first implementation.
reduction_kind = "rmsnorm_like"

# Fusion scope = number of semantic stages fused around the reduction.
fusion_scopes = [1, 2, 3, 4, 5, 6]

# Runtime config.
block = 256
warmup = 20
repeat = 100
trials = 5
check_n = 4096
output_csv = "results/reduction_elementwise_postprocessing.csv"

expected_curve = "execution time decreases first, then plateaus or rebounds when reduction-side resource pressure dominates"
+++

# Reduction + Elementwise Post-processing

## DAG

This motif studies a reduction followed by normalization or elementwise post-processing.

The first implementation can use an RMSNorm-like graph:

```text
x
├── x^2 -> reduce_sum over hidden dimension -> mean -> rsqrt(mean + eps)
└───────────────────────────────────────────────────────────────┐
                                                                v
                                    x * rsqrt(mean + eps) * gamma + bias -> y
```

Equivalent formula:

```text
r[row] = rsqrt(mean_col(x[row, col]^2) + eps)
y[row, col] = x[row, col] * r[row] * gamma[col] + bias[col]
```

A softmax-like variant can be added later, but RMSNorm-like reduction is simpler for the first generator.

## LLM relevance

This motif abstracts common normalization and reduction-heavy operators in LLM blocks:

```text
RMSNorm before Attention
RMSNorm before MLP
LayerNorm variants
Softmax-like reduction followed by elementwise normalization
```

These operators frequently sit on fusion boundaries between larger compute kernels.

## Fusion scope

`fusion_group_size` means how many semantic stages around the reduction are fused.

Suggested stage mapping:

- `G=1`: reduction statistics and post-processing are mostly separated.
- `G=2`: fuse square + reduce_sum.
- `G=3`: fuse square + reduce_sum + mean/rsqrt.
- `G=4`: additionally fuse normalization multiply `x * r`.
- `G=5`: additionally fuse gamma scale.
- `G=6`: fuse the full RMSNorm-like post-processing, including bias/residual-style elementwise output.

The exact kernel decomposition can be adjusted by the generator, but the meaning should remain:

```text
larger G = reduction is fused with more nearby elementwise post-processing
```

## Expected behavior

Small-scope fusion should reduce intermediate tensors such as squared values, row-wise statistics, and normalized outputs. However, reduction kernels already consume more local resources than simple elementwise kernels.

Possible over-fusion causes:

```text
reduction buffer pressure
more live values per element
higher register / UB usage
more complex synchronization or staged reduction logic
larger instruction footprint
```

The expected curve is decreasing first, then plateauing or rebounding when the cost of fusing more post-processing exceeds the saved memory traffic.

## Implementation notes for the generator

The first implementation should use fixed `default_cols=256` so that one row reduction can be handled in a controlled way. The generator can later add larger hidden sizes or multi-block reductions as stress tests.

Correctness should be checked against a PyTorch reference. The benchmark should report only execution time.
