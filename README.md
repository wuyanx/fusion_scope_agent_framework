# Fusion Scope Agent Framework

This project generalizes the first Ascend 910B fusion-scope microbenchmark into a scenario-driven, agent-friendly framework.

The original validated line was:

- hardware: Ascend 910B
- scenario: multi-branch elementwise + branch-wise softmax aggregation
- metric: execution time only
- variable: `fusion_group_size`
- result: execution time monotonically decreased for `B=32`, with smaller marginal gain after `G=8`

This framework keeps that line reproducible and adds a generic path for adding more motifs.

## Core idea

Each fusion scenario is described by a human-readable Markdown file with TOML front matter:

```text
scenarios/*.md
```

The front matter contains machine-readable fields such as:

```toml
id = "02_branch_fan_in_aggregation"
benchmark_kind = "branch_fan_in_aggregation"
default_N = 4194304
default_B = 32
fusion_scopes = [1, 2, 4, 8, 16, 32]
block = 256
warmup = 20
repeat = 100
trials = 5
```

An agent or user can then run:

```bash
python tools/run_scenario.py \
  --scenario scenarios/02_branch_fan_in_aggregation.md \
  --overwrite \
  --dry-run
```

or on the Ascend server:

```bash
/root/miniconda3/envs/tlx/bin/python tools/run_scenario.py \
  --scenario scenarios/02_branch_fan_in_aggregation.md \
  --overwrite \
  --check \
  --run \
  --plot \
  --device npu:0
```

The command will:

1. read the scenario file;
2. materialize a generated benchmark project under `generated/<scenario_id>/`;
3. run correctness check if requested;
4. run benchmark if requested;
5. generate plots if requested.

## Current supported scenarios

| Scenario file | Motif | Fusion scope definition |
|---|---|---|
| `scenarios/01_linear_elementwise_chain.md` | Linear elementwise chain | number of consecutive elementwise ops fused into one kernel |
| `scenarios/02_branch_fan_in_aggregation.md` | Branch fan-in aggregation | number of branches fused into one group kernel |

## Directory layout

```text
.
├── README.md
├── AGENTS.md
├── CODEX_HANDOFF_PROMPT.md
├── scenarios/
│   ├── 01_linear_elementwise_chain.md
│   └── 02_branch_fan_in_aggregation.md
├── fusion_scope_core/
│   ├── scenario.py
│   └── file_utils.py
├── generators/
│   ├── linear_elementwise_chain.py
│   ├── branch_fan_in_aggregation.py
│   └── common.py
├── tools/
│   ├── materialize_scenario.py
│   ├── run_scenario.py
│   ├── remote_run_scenario.py
│   ├── plot_results.py
│   └── check_env.py
├── scripts/
│   └── connect_910b3.py
└── skills/
    └── fusion_scope_framework/
        └── SKILL.md
```

## Local dry run

Local machine may not have Ascend/Triton. Use dry run to test parsing and code generation:

```bash
python tools/run_scenario.py \
  --scenario scenarios/01_linear_elementwise_chain.md \
  --overwrite \
  --dry-run
```

```bash
python tools/run_scenario.py \
  --scenario scenarios/02_branch_fan_in_aggregation.md \
  --overwrite \
  --dry-run
```

## Server run

If your `~/.ssh/config` already has a `910B3` alias:

```bash
python tools/remote_run_scenario.py \
  --host 910B3 \
  --remote-root /home/wyx/fusion_scope_agent_framework \
  --python /root/miniconda3/envs/tlx/bin/python \
  --scenario scenarios/02_branch_fan_in_aggregation.md \
  --overwrite \
  --check \
  --run \
  --plot
```

This uses `rsync` and `ssh`, so it is intentionally simple and compatible with normal agent workflows.

## Generated benchmark structure

For each scenario, the framework creates:

```text
generated/<scenario_id>/
├── scenario.md
├── README_GENERATED.md
├── run_benchmark.py
├── plot_results.py
└── fusion_benchmark/
    ├── __init__.py
    ├── device.py
    ├── kernels.py
    └── torch_reference.py
```

The generated benchmark can be copied and run independently.

## Important Ascend compatibility note

The first validated benchmark encountered Triton-Ascend/CANN 9 beta issues around masked load/store and `bufferization.to_tensor`. Therefore the generated kernels use unmasked load/store and require:

```text
N % block == 0
check_n % block == 0
```

This is intentional for the current environment. Do not remove this constraint unless the server stack is upgraded and the kernels are revalidated.

## Extension path

To add a new motif:

1. Add a new `scenarios/XX_name.md` file with TOML front matter.
2. Add a generator under `generators/<benchmark_kind>.py`.
3. Register it in `generators/__init__.py`.
4. Run `python tools/run_scenario.py --scenario ... --dry-run`.
5. Run correctness check and benchmark on Ascend.

First-stage planned motif expansion after these two:

- GLU / SwiGLU gating
- Elementwise + layout transform
- Reduction + elementwise post-processing
- Cascaded reduction
- Quant / dequant / cast interleaving
