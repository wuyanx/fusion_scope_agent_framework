# Fusion Scope Framework Skill

Use this skill when the user asks you to run, debug, extend, or summarize the Ascend 910B fusion-scope motif benchmark framework.

## Objective

The framework studies how execution time changes as `fusion_group_size` increases for a fixed operator DAG motif on Ascend 910B.

The metric is execution time only.

## Standard scenario workflow

1. Read the target scenario file under `scenarios/*.md`.
2. Parse the TOML front matter.
3. Classify scenario state:

```bash
python tools/agent_run_scenario.py --scenario scenarios/<file>.md --status-only
```

4. Act on the state:

- `existing_unchanged`: tell the user the scenario already has up-to-date generated scripts, then run the existing generated benchmark.
- `existing_modified`: tell the user the scenario file changed, then regenerate scripts before running.
- `new_scenario_existing_benchmark`: tell the user this is a new scenario using an existing benchmark kind, then materialize scripts before running.
- `new_benchmark_kind`: tell the user this is a new benchmark kind, then create `generators/<benchmark_kind>.py`, register it in `generators/__init__.py`, dry-run it, and only then run on Ascend.

5. For a local dry-run or server run, prefer the agent-facing runner:

```bash
python tools/agent_run_scenario.py --scenario scenarios/<file>.md --dry-run
```

On the Ascend server:

```bash
/root/miniconda3/envs/tlx/bin/python tools/run_scenario.py \
  --scenario scenarios/<file>.md \
  --overwrite \
  --check \
  --run \
  --plot \
  --device npu:0
```

or:

```bash
/root/miniconda3/envs/tlx/bin/python tools/agent_run_scenario.py \
  --scenario scenarios/<file>.md \
  --check \
  --run \
  --plot \
  --device npu:0
```

6. Collect CSV and plots from `generated/<scenario_id>/results/<timestamp>/`.

## Current supported benchmark kinds

- `linear_elementwise_chain`
- `branch_fan_in_aggregation`

## New benchmark kind responsibilities

When `tools/agent_run_scenario.py --status-only` reports `new_benchmark_kind`, do not ask the user to create or register generator code. The user-facing contract is that the user writes the scenario and asks the agent to test it; the agent completes framework code.

Required agent actions:

1. Read the scenario Markdown carefully and preserve its DAG and `fusion_group_size` definition.
2. Add `generators/<benchmark_kind>.py`.
3. Reuse `generators/common.py` templates where possible.
4. Add a deterministic PyTorch reference implementation.
5. Add Triton kernels with current Ascend compatibility constraints: unmasked load/store and `N % block == 0`.
6. Register the generator in `generators/__init__.py`.
7. Run `python tools/agent_run_scenario.py --scenario scenarios/<file>.md --dry-run`.
8. Run correctness check and benchmark on Ascend.

If the scenario describes a motif that cannot be implemented safely from the Markdown alone, ask one concise clarifying question. Otherwise make conservative implementation choices and continue.

## Invariants

- Do not change a scenario's DAG unless the user explicitly asks.
- Do not change the meaning of `fusion_group_size`.
- Do not add profiler metrics unless requested.
- Keep PyTorch reference checks for correctness.
- Generated kernels currently use unmasked load/store for Ascend CANN 9 beta compatibility. Preserve `N % block == 0` checks.
- If a previously generated scenario file changes, regenerate the generated benchmark before running so the copied scenario, config, and scripts are in sync.

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
result_dir:
short explanation:
```
