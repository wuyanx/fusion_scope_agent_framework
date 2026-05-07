# Prompt for Codex CLI Agent

你现在接手的是一个 Ascend 910B fusion-scope benchmark 框架项目，目录名是 `fusion_scope_agent_framework`。

项目目标：把已经跑通的第一个 benchmark（multi-branch elementwise + branch-wise softmax aggregation）扩展成一个“scenario-driven”的可复现分析框架。以后每一种 fusion motif 都用一个 `scenarios/*.md` 文件描述，agent 读取场景文件后生成对应 Triton benchmark，上传/运行到 Ascend 910B 服务器，最后输出 execution time 曲线。

请先阅读：

- `README.md`
- `AGENTS.md`
- `scenarios/01_linear_elementwise_chain.md`
- `scenarios/02_branch_fan_in_aggregation.md`
- `skills/fusion_scope_framework/SKILL.md`

当前第一阶段只支持两个 scenario：

1. Linear Elementwise Chain
   - DAG: `x -> op1 -> op2 -> ... -> y`
   - fusion_group_size = 一个 kernel 内连续融合的 elementwise op 数

2. Branch Fan-in Aggregation
   - DAG: `x -> 多分支 -> branch-wise softmax aggregation -> y`
   - fusion_group_size = 一个 group kernel 内融合的 branch 数

先在本地做 dry-run：

```bash
python tools/run_scenario.py --scenario scenarios/01_linear_elementwise_chain.md --overwrite --dry-run
python tools/run_scenario.py --scenario scenarios/02_branch_fan_in_aggregation.md --overwrite --dry-run
```

然后在 Ascend 910B 服务器上运行：

```bash
/root/miniconda3/envs/tlx/bin/python tools/run_scenario.py \
  --scenario scenarios/01_linear_elementwise_chain.md \
  --overwrite \
  --check \
  --run \
  --plot \
  --device npu:0
```

```bash
/root/miniconda3/envs/tlx/bin/python tools/run_scenario.py \
  --scenario scenarios/02_branch_fan_in_aggregation.md \
  --overwrite \
  --check \
  --run \
  --plot \
  --device npu:0
```

如果需要从本地同步到服务器，可以使用：

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

注意事项：

- 第一阶段只看 execution time。
- 不要引入 profiler 指标。
- 不要改变现有 scenario 的数学 DAG。
- 不要改变现有 `fusion_group_size` 的定义。
- 当前 Ascend Triton/CANN 9 beta 路径下，generated kernels 使用 unmasked load/store，因此 `N` 和 `check_n` 必须能被 `block` 整除。
- 如果遇到 Triton/CANN 编译兼容性问题，只做最小修复，保持数学形式和实验定义不变。

完成后请总结：

- 环境信息；
- scenario id；
- N、problem_size、fusion scopes、block、warmup/repeat/trials；
- 每个 fusion_group_size 的 execution_time_ms；
- 最优 fusion_group_size；
- 曲线形态和简要解释。
