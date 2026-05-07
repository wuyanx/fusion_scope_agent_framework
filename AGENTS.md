# Agent Instructions

You are working on a scenario-driven Ascend 910B fusion-scope benchmark framework.

## Hard constraints

1. The first-stage performance metric is execution time only.
2. Do not add bandwidth, occupancy, register, profiler, or roofline metrics unless explicitly requested.
3. Do not change the meaning of `fusion_group_size` for an existing scenario.
4. Do not silently change the operator graph of an existing scenario.
5. Do not expand this into a full compiler fusion strategy project. This framework is for reproducible motif-level microbenchmarks.
6. Current Ascend Triton/CANN 9 beta compatibility requires unmasked load/store and `N % block == 0`.

## Workflow

For any scenario request:

1. Read the target `scenarios/*.md` file.
2. Check the TOML front matter.
3. Materialize the benchmark:

```bash
python tools/run_scenario.py --scenario <scenario.md> --overwrite --dry-run
```

4. On Ascend server, run:

```bash
/root/miniconda3/envs/tlx/bin/python tools/run_scenario.py \
  --scenario <scenario.md> \
  --overwrite \
  --check \
  --run \
  --plot \
  --device npu:0
```

5. Report:

- environment;
- scenario id and benchmark kind;
- N, block, warmup, repeat, trials;
- fusion scopes;
- execution_time_ms table;
- best fusion_group_size;
- curve shape: monotonic decrease / plateau / rebound.

## Adding a new scenario

Add both:

- `scenarios/XX_new_motif.md`
- `generators/new_motif.py`

Register the generator in `generators/__init__.py`.

Keep generated code self-contained and easy to inspect. Prefer simple deterministic coefficients and a PyTorch reference implementation for correctness checks.

## Server details known from previous validated run

- server path used before: `/home/wyx/ascend_fusion_scope_benchmark`
- recommended new path: `/home/wyx/fusion_scope_agent_framework`
- Python: `/root/miniconda3/envs/tlx/bin/python`
- torch: `2.7.1+cpu`
- torch_npu: `2.7.1`
- triton: `3.5.0`
- device: `npu:0`

If compiler issues appear in CANN 9 beta, only make minimal compatibility fixes and preserve the scenario math.
