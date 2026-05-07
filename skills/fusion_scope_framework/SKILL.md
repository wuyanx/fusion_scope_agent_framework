# Fusion Scope Framework Skill

Use this skill when the user asks you to run, debug, extend, or summarize the Ascend 910B fusion-scope motif benchmark framework.

## Objective

The framework studies how execution time changes as `fusion_group_size` increases for a fixed operator DAG motif on Ascend 910B.

The metric is execution time only.

## Standard scenario workflow

1. Read the target scenario file under `scenarios/*.md`.
2. Parse the TOML front matter.
3. Materialize a generated benchmark:

```bash
python tools/run_scenario.py --scenario scenarios/<file>.md --overwrite --dry-run
```

4. On the Ascend server, run:

```bash
/root/miniconda3/envs/tlx/bin/python tools/run_scenario.py \
  --scenario scenarios/<file>.md \
  --overwrite \
  --check \
  --run \
  --plot \
  --device npu:0
```

5. Collect CSV and plots from `generated/<scenario_id>/results/`.

## Current supported benchmark kinds

- `linear_elementwise_chain`
- `branch_fan_in_aggregation`

## Invariants

- Do not change a scenario's DAG unless the user explicitly asks.
- Do not change the meaning of `fusion_group_size`.
- Do not add profiler metrics unless requested.
- Keep PyTorch reference checks for correctness.
- Generated kernels currently use unmasked load/store for Ascend CANN 9 beta compatibility. Preserve `N % block == 0` checks.

## Output summary template

When reporting a run, include:

```text
scenario_id:
benchmark_kind:
hardware/device:
N:
problem_size:
fusion_scopes:
block/warmup/repeat/trials:

fusion_group_size | num_kernel_launches | execution_time_ms
...

best fusion_group_size:
curve shape:
short explanation:
```
