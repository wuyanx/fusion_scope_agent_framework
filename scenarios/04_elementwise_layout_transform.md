+++
id = "04_elementwise_layout_transform"
name = "Elementwise + Layout Transform"
benchmark_kind = "elementwise_layout_transform"
hardware = "Ascend 910B"

# Problem size.
# The default 2D tensor has default_rows * default_cols elements.
default_rows = 4096
default_cols = 1024

# Elementwise ops placed before and after the layout transform.
default_pre_ops = 8
default_post_ops = 8

# Fusion scope = number of elementwise ops fused around one layout-transform kernel.
# The layout transform itself is always included in the central transform stage.
fusion_scopes = [1, 2, 4, 8, 16]

# Runtime config.
block = 256
warmup = 20
repeat = 100
trials = 5
check_n = 4096
output_csv = "results/elementwise_layout_transform.csv"

expected_curve = "execution time may decrease first, then rebound when fused layout access becomes too complex"
+++

# Elementwise + Layout Transform

## DAG

This motif studies elementwise computation around a data-layout-changing operation:

```text
x
-> pre_elementwise_1
-> pre_elementwise_2
-> ...
-> reshape / transpose / concat / layout_convert
-> post_elementwise_1
-> post_elementwise_2
-> ...
-> y
```

A concrete benchmark can use a 2D transpose as the layout transform:

```text
x[row, col]
-> pre elementwise chain
-> transpose
-> post elementwise chain
-> y[col, row]
```

## LLM relevance

Layout transforms are common in LLM execution graphs, for example:

```text
QKV projection output
-> split q/k/v
-> reshape to [batch, seq, heads, head_dim]
-> transpose to attention-friendly layout
```

and:

```text
attention output
-> transpose
-> reshape / concat heads
-> output projection
```

RoPE and other elementwise transformations may also appear before or after Q/K layout transforms.

## Fusion scope

`fusion_group_size` means the number of elementwise operations fused around the layout transform.

The generator can interpret this as:

```text
G = total number of pre/post elementwise ops included in the same Triton kernel as the layout transform
```

Examples:

- `G=1`: only one adjacent elementwise op is fused with the layout transform; most pre/post ops remain separate.
- `G=4`: two pre-transform and two post-transform elementwise ops are fused with the transform.
- `G=16`: all 8 pre-ops, the layout transform, and all 8 post-ops are fused as much as possible.

For odd `G`, the generator may assign `floor(G/2)` ops before the transform and `ceil(G/2)` ops after the transform, or document its chosen convention.

## Expected behavior

This motif is more likely than a pure elementwise chain to show a performance turning point. Fusion reduces intermediate writes around the layout transform, but it also combines arithmetic with strided reads/writes and more complicated address calculation.

Potential over-fusion causes:

```text
strided memory access
more complicated index arithmetic
lower cache / UB locality
bank conflict or memory pipeline pressure
larger instruction footprint
```

Therefore the expected curve may decrease initially and then rebound if the fused layout-transform kernel becomes too complex.

## Implementation notes for the generator

The first implementation can use a deterministic 2D transpose benchmark:

```text
input:  x[default_rows, default_cols]
output: y[default_cols, default_rows]
```

The pre/post elementwise chain can use the same non-trivial transform style as the linear elementwise benchmark:

```text
z = tanh(a_i * z + b_i) + 0.125 * sigmoid(c_i * z + d_i) + 0.03125 * z
```

Correctness should be checked against a PyTorch reference. The benchmark should report only execution time.
