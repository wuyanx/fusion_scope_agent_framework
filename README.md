# Fusion Scope Agent Framework

This project is a scenario-driven benchmark framework for studying how fusion scope affects execution time on Ascend 910B. It focuses on reproducible motif-level microbenchmarks: each scenario describes one operator DAG motif, a definition of `fusion_group_size`, and the runtime configuration needed to generate execution-time curves.

The first-stage metric is execution time only. The framework is intentionally scoped to motif-level experiments rather than profiler analysis, roofline modeling, or full compiler fusion strategy exploration.

## Usage

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

The Markdown body records the human-facing definition of the motif: the DAG shape, where it appears in LLM workloads, what `fusion_group_size` means for that scenario, and what curve shape is expected.

If `scenarios/` already contains the scenario you want to test, you can use that file directly. If you want to change an experiment or add a new motif, edit an existing scenario file or add a new `scenarios/XX_name.md` following the same format.

Then tell the agent:

```text
Please test scenarios/XX_name.md
```

From the user's perspective, the scenario Markdown file is the only required input. The agent handles the rest:

1. Read the scenario and parse the TOML front matter.
2. Classify the scenario:
   - if the scenario already exists and has not changed, reuse the existing generated benchmark;
   - if the scenario is new or the Markdown file changed, materialize a fresh generated benchmark;
   - if the scenario uses a new `benchmark_kind`, create the required generator under `generators/`, register it, and then materialize the benchmark.
3. Run local dry-run checks when appropriate.
4. Run correctness checks and benchmarks on the Ascend server.
5. Generate CSV output and execution-time plots under `generated/<scenario_id>/results/<timestamp>/`.
6. Summarize the environment, scenario config, execution-time table, best `fusion_group_size`, and curve shape.

The user does not need to manually run framework commands, edit `generators/__init__.py`, or manage generated benchmark scripts. If a new scenario does not describe the DAG or math clearly enough to implement safely, the agent should ask one concise clarification before generating code.

The agent-facing scenario states are:

- `existing_unchanged`: generated scripts are up to date; run the existing generated benchmark.
- `existing_modified`: the scenario file changed; regenerate generated scripts before running.
- `new_scenario_existing_benchmark`: the `benchmark_kind` is supported, but this scenario has not been materialized yet; generate scripts before running.
- `new_benchmark_kind`: the scenario needs a new generator and registry entry; the agent creates those framework files, then runs the benchmark workflow.

## Directory layout

```text
.
├── README.md                         # Project overview and expected workflow.
├── AGENTS.md                         # Operating rules and hard constraints for agents.
├── requirements.txt                  # Lightweight local requirements; server uses its Ascend environment.
├── scenarios/                        # Scenario specs: Markdown + TOML front matter.
│   ├── 01_linear_elementwise_chain.md
│   └── 02_branch_fan_in_aggregation.md
├── fusion_scope_core/                # Shared parser and file utilities.
│   ├── scenario.py
│   └── file_utils.py
├── generators/                       # Benchmark materializers, one per benchmark_kind.
│   ├── __init__.py                   # Registry from benchmark_kind to generator.
│   ├── common.py                     # Shared generated project templates.
│   ├── linear_elementwise_chain.py
│   └── branch_fan_in_aggregation.py
├── tools/                            # Local and remote workflow entry points.
│   ├── agent_run_scenario.py
│   ├── materialize_scenario.py
│   ├── run_scenario.py
│   ├── remote_run_scenario.py
│   ├── plot_results.py
│   └── check_env.py
├── scripts/                          # Convenience scripts for known server setups.
│   └── connect_910b3.py
├── skills/                           # Agent-side workflow guidance.
│   └── fusion_scope_framework/
│       └── SKILL.md
└── generated/                        # Generated benchmark projects and run artifacts.
    └── <scenario_id>/
        ├── scenario.md               # Copy of the source scenario used for this benchmark.
        ├── README.md                 # Generated benchmark notes.
        ├── README_GENERATED.md       # Compatibility copy of the generated README.
        ├── run_benchmark.py          # Self-contained correctness and benchmark runner.
        ├── plot_results.py           # Plot helper for the generated CSV.
        ├── fusion_benchmark/         # Generated benchmark package.
        │   ├── __init__.py
        │   ├── device.py             # Device resolution, synchronization, timing helpers.
        │   ├── kernels.py            # Triton kernels and FusionPlan implementation.
        │   └── torch_reference.py    # PyTorch reference for correctness checks.
        └── results/                  # Per-run CSV output and execution-time plots.
            └── <timestamp>/          # Format: YYYYMMDD_HHMMSS_microseconds.
                ├── <scenario>.csv
                ├── fusion_scope_execution_time.png
                └── fusion_scope_speedup.png
```

Generated benchmark directories are self-contained enough to inspect, copy, and run on the Ascend server. They are artifacts of the scenario workflow, not hand-maintained source files.

## Environment

The usual remote defaults live in `tools/remote_run_scenario.py`:

```text
host = 910B3
remote_root = /home/wyx/fusion_scope_agent_framework
python = /root/miniconda3/envs/tlx/bin/python
device = npu:0
```

When running directly on the server, the Python executable and device come from the command the agent runs. The expected server stack is the Ascend environment with `torch`, `torch_npu`, Triton, and NPU access available.

## Important Ascend compatibility note

The first validated benchmark encountered Triton-Ascend/CANN 9 beta issues around masked load/store and `bufferization.to_tensor`. Therefore the generated kernels use unmasked load/store and require:

```text
N % block == 0
check_n % block == 0
```

This is intentional for the current environment. Do not remove this constraint unless the server stack is upgraded and the kernels are revalidated.
