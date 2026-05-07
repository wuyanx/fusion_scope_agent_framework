+++
id = "06_cascaded_reduction"
name = "Cascaded Reduction"
benchmark_kind = "cascaded_reduction"
hardware = "Ascend 910B"

# Problem size.
# The default tensor is interpreted as [default_rows, default_cols].
default_rows = 8192
default_cols = 512

# Use a tile-local two-level softmax-like reduction for the first implementation.
reduction_kind = "tile_softmax_like"
default_tile_cols = 128
default_reduction_rounds = 4

# Fusion scope = number of reduction rounds / reduction-adjacent stages fused together.
fusion_scopes = [1, 2, 3, 4]

# Runtime config.
block = 256
warmup = 20
repeat = 100
trials = 5
check_n = 4096
output_csv = "results/cascaded_reduction.csv"

expected_curve = "U-shaped behavior is more likely because multi-stage reduction increases local buffer and synchronization pressure"
+++

# Cascaded Reduction

## DAG

This motif studies multi-stage reductions, where the final result is obtained through several reduction rounds.

A softmax-like cascaded reduction can be abstracted as:

```text
x
-> partial max
-> second-level max
-> exp(x - global_max)
-> partial sum
-> second-level sum
-> normalize
-> y
```

For the first implementation, this should be kept tile-local rather than requiring real cross-block global synchronization.

A controlled tile-local version can use:

```text
x[row, col]
-> split each row into tiles
-> tile-level max
-> row-level max over tile results
-> exp(x - row_max)
-> tile-level sum
-> row-level sum over tile results
-> y = exp(x - row_max) / row_sum
```

## LLM relevance

This motif appears in reduction-heavy LLM operations when one reduction dimension is too large or naturally decomposed into multiple levels:

```text
large softmax
2D softmax variants
hierarchical pooling / token merging
large hidden-size normalization variants
multi-stage score aggregation
```

It is also useful as a stress version of the simpler reduction + post-processing motif.

## Fusion scope

`fusion_group_size` means the number of reduction rounds or reduction-adjacent stages fused together.

Suggested interpretation:

- `G=1`: partial reductions, second-level reductions, and normalization are separate.
- `G=2`: fuse adjacent pairs, such as partial max + second-level max, or partial sum + second-level sum.
- `G=3`: fuse one complete max side or sum side with its adjacent elementwise transform.
- `G=4`: fuse the full tile-local cascaded reduction and normalization as much as possible.

The key idea is:

```text
larger G = more reduction stages are handled by one fused kernel or one tightly connected fused group
```

## Expected behavior

This scenario is more likely to show a U-shaped curve than simple elementwise chains because the fused kernel may need to keep more local reduction state.

Potential over-fusion causes:

```text
tile-level reduction buffers
second-level reduction buffers
exp / sum / normalize temporary values
shared-memory or UB pressure
register pressure
synchronization and scheduling complexity
```

However, a true global multi-block reduction may require multiple kernels. The first benchmark should avoid unsupported global synchronization by using a tile-local or row-local design that is reproducible on Ascend 910B.

## Implementation notes for the generator

The first generator can implement a row-wise softmax-like operation with a fixed reduction width and internal tile partitioning. If the full cascaded version is too difficult on the current Triton-Ascend stack, implement a simplified two-stage row-local reduction and document the simplification in the generated README.

Correctness should be checked against a PyTorch reference. The benchmark should report only execution time.
