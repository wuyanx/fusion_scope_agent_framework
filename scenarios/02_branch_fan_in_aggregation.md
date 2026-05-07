+++
id = "02_branch_fan_in_aggregation"
name = "Branch Fan-in Aggregation"
benchmark_kind = "branch_fan_in_aggregation"
hardware = "Ascend 910B"

# Problem size.
default_N = 4194304
default_B = 32

# Fusion scope = number of branches fused into one Triton group kernel.
fusion_scopes = [1, 2, 4, 8, 16, 32]

# Runtime config.
block = 256
warmup = 20
repeat = 100
trials = 5
check_n = 4096
output_csv = "results/branch_fan_in_aggregation.csv"

expected_curve = "execution time decreases first, then plateaus, and may rebound under stress settings"
+++

# Branch Fan-in Aggregation

## DAG

```text
                 branch_0: score_0, value_0
               /
x ------------ branch_1: score_1, value_1
               \
                ...
                 branch_{B-1}: score_{B-1}, value_{B-1}

-> branch-wise softmax aggregation
-> y
```

For every input element `x[n]`, the generated benchmark computes `B` branch scores and values, then performs a branch-wise softmax weighted sum:

```text
weight_i = exp(score_i) / sum_j exp(score_j)
y = sum_i weight_i * value_i
```

## LLM relevance

This motif abstracts gating, router score processing, multi-path mixing, and branch fan-in structures in LLM operator DAGs.

## Fusion scope

`fusion_group_size` means the number of branches fused into one Triton group kernel.

Examples:

- `G=1`: each branch is processed by one group kernel; final combine kernel merges all branch groups.
- `G=8`: every 8 branches are processed by one group kernel; final combine merges 4 groups.
- `G=32`: all branches and the final aggregation are fused into a single kernel.

## Expected behavior

The initial validated run on Ascend 910B with `B=32` showed monotonic execution-time decrease, with smaller marginal gain after `G=8`. Stress settings such as `B=64/128` or more complex branch math may be needed to observe a clear over-fusion rebound.
