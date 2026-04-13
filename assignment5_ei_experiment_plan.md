# Assignment 5 第 5 章 EI 实验计划

## 1. 目标与验收标准

基于 [cs336_spring2025_assignment5_alignment_zh.md](/home/nic/wyz/assignment5-alignment/cs336_spring2025_assignment5_alignment_zh.md) 第 5 章与当前仓库实现，EI 阶段的目标不是只“把代码补完”，而是完成一条可复现的 `generate -> filter -> retrain -> evaluate` 闭环，并拿到可写进报告的结果。

本阶段的硬性验收标准：

- 使用 `Qwen2.5-Math-1.5B` base 模型，在 MATH `train.jsonl` 上运行 EI。
  文档中的标准路径是 `/data/a5-alignment/MATH/train.jsonl`，如果本地仓库做了数据映射，也可在入口里覆写为仓库内路径。
- 固定 `n_ei_steps = 5`。
- 每个 EI step 的问题批量大小 `Db` 至少覆盖 `{512, 1024, 2048}`。
- 至少尝试 2 组不同的 rollout 次数 `G` 与 SFT epoch 数。
- 记录验证准确率曲线。
- 记录训练过程中模型响应熵曲线。
- 产出一个在 MATH validation 上至少达到 `15%` `avg_answer_reward` 的模型。
- 写出 2 句话，对比 EI 与 SFT，以及不同 EI step 之间的表现。

## 2. 当前代码现状与可复用组件

### 已有能力

- [cs336_alignment/sft.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/sft.py) 已经有完整的 SFT 训练主循环 `run_sft_experiment`，支持：
  - 训练/验证拆分
  - `history.json` 持久化
  - `best_checkpoint` / `final_checkpoint`
  - 本地 smoke 与 cloud profile
  - vLLM 周期性验证
- [cs336_alignment/sft.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/sft.py) 里已有 `_filter_correct_sft_examples`，说明“按 `answer_reward == 1.0` 过滤正确轨迹再训练”的思路已经在 SFT 阶段落地过。
- [cs336_alignment/rewards.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/rewards.py) 与 [cs336_alignment/drgrpo_grader.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/drgrpo_grader.py) 已具备 EI 所需的 verified reward。
- [cs336_alignment/logging_utils.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/logging_utils.py) 已能统计 `mean_token_entropy`、`response_length`、若干样例日志，这非常适合直接接入 EI 的中间评测。

### 当前缺口

- [cs336_alignment/expert_iteration.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/expert_iteration.py) 已经有可运行的 EI 主循环与 `single` CLI，但还没有 sweep 子命令。
- [scripts/run_expert_iteration.py](/home/nic/wyz/assignment5-alignment/scripts/run_expert_iteration.py) 已经接到新的 CLI，可直接运行 `single` 命令。
- EI 专属指标目前主要写到本地 `summary.json` / `step_summary.json`，还没有完整接入 wandb 面板。

结论：当前 EI 的实现方式不是重写一套全新 trainer，而是复用 `sft.py` 的训练/评测骨架，把 EI 看成“离线生成新 SFT 数据，再调用 SFT 训练器”的 5 轮数据闭环。

## 3. 推荐的实现路径

### 阶段 A：先跑最小可运行 EI 闭环

目标：先跑通 `1` 个 EI step 的 smoke，不追求分数。

当前脚本已经具备的核心能力：

- 在 `expert_iteration.py` 中读取 `train.jsonl` 问题与标准答案。
- 用 `r1_zero.prompt` 构造 prompt。
- 用 vLLM 对每个问题采样 `G` 条 rollout。
- `SamplingParams` 需要显式设置：
  - `temperature=1.0`
  - `top_p=1.0`
  - `max_tokens=1024`
  - `min_tokens=4`
  - `stop=["</answer>"]`
  - `include_stop_str_in_output=True`
- 用 `r1_zero_reward_fn(response, ground_truth)` 过滤出 `answer_reward == 1.0` 的轨迹。
- 把过滤后的样本写成 EI step 专用的 SFT 数据文件，每条样本至少包含：
  - `prompt`
  - `response`
  - `ground_truth`
- 调用现有 SFT 风格训练逻辑，对这个 step 的“专家轨迹数据”训练若干 epoch。
- 在 step 结束后，用 validation 集评测并保存：
  - `avg_answer_reward`
  - `avg_reward`
  - `avg_format_reward`
  - 过滤前 rollout 总数
  - 过滤后保留样本数
  - 保留率
  - 响应熵均值

### 阶段 B：把 smoke 扩成正式 5-step 实验

目标：完成第 5 章要求的 `n_ei_steps=5` 正式实验。

建议每个 EI step 固定做下面 6 件事：

1. 从当前 policy 生成当前 step 的 rollout。
2. 过滤正确轨迹，得到 step 数据集。
3. 记录过滤后数据量与保留率。
4. 在 step 数据集上做 SFT 微调。
5. 在 validation 上做评测。
6. 记录该 step 的响应熵与 checkpoint。

### 阶段 C：整理报告导出与画图输入

最终应该能稳定导出：

- 顶层 `summary.json`
- 每个 step 的 `step_summary.json`
- 每个 EI step 的 `history.json`
- 每个 EI step 的 `best_checkpoint`
- 用于画验证准确率曲线的数据
- 用于画熵曲线的数据

## 4. 当前脚本入口

EI 当前可直接使用的命令入口是：

```bash
uv run python scripts/run_expert_iteration.py single [OPTIONS]
```

最常用的参数如下：

- `--profile local-smoke|cloud`
- `--output-dir PATH`
- `--train-device cuda:0`
- `--rollout-device cuda:1`
- `--eval-device cuda:1|none`
- `--ei-batch-size INT`
- `--rollouts-per-question INT`
- `--num-epochs INT`
- `--n-ei-steps INT`
- `--eval-num-examples INT`
- `--entropy-num-examples INT`
- `--use-wandb`
- `--wandb-project NAME`
- `--wandb-run-name NAME`

## 5. 实验矩阵

### 5.1 Smoke 配置

用途：确认实现正确，不浪费 GPU。

- `Db = 64`
- `G = 2`
- `num_epochs = 1`
- `n_ei_steps = 1`
- `eval_num_examples = 64`
- 只检查：
  - 不会生成空串
  - 过滤后样本不为 0
  - 能完成一次训练与一次验证
  - 能落盘 history 与 step summary

对应命令：

```bash
uv run python scripts/run_expert_iteration.py single \
  --profile local-smoke \
  --train-device cuda:0 \
  --rollout-device cuda:1 \
  --eval-device none \
  --output-dir /root/autodl-tmp/assignment5-assets/results/expert_iteration/smoke
```

### 5.2 正式主实验

用途：满足作业要求并拿到结论。

建议先跑 2 组主配置，再视结果扩展：

| 配置名 | Db | G | EI 内 SFT epochs | n_ei_steps | 目的 |
| --- | --- | --- | --- | --- | --- |
| EI-A | 512 | 4 | 1 | 5 | 最保守主线，先看 EI 是否稳定提升 |
| EI-B | 1024 | 4 | 2 | 5 | 固定较小 rollout，单独观察更多训练是否有帮助 |
| EI-C | 2048 | 8 | 2 | 5 | 同时提高 rollout 与训练强度，追求最好分数 |

建议执行顺序：

1. 先跑 `EI-A`，确认 5 step 趋势正常。
2. 再跑 `EI-B`，观察“更多训练”是否优于 `EI-A`。
3. 最后跑 `EI-C`，观察“更大 rollout”是否能带来更高质量专家数据。

对应命令：

`EI-A`

```bash
uv run python scripts/run_expert_iteration.py single \
  --profile cloud \
  --train-device cuda:0 \
  --rollout-device cuda:1 \
  --eval-device cuda:1 \
  --ei-batch-size 512 \
  --rollouts-per-question 4 \
  --num-epochs 1 \
  --n-ei-steps 5 \
  --output-dir /root/autodl-tmp/assignment5-assets/results/expert_iteration/ei_a \
  --use-wandb \
  --wandb-project cs336-alignment-ei \
  --wandb-run-name ei-a
```

`EI-B`

```bash
uv run python scripts/run_expert_iteration.py single \
  --profile cloud \
  --train-device cuda:0 \
  --rollout-device cuda:1 \
  --eval-device cuda:1 \
  --ei-batch-size 1024 \
  --rollouts-per-question 4 \
  --num-epochs 2 \
  --n-ei-steps 5 \
  --output-dir /root/autodl-tmp/assignment5-assets/results/expert_iteration/ei_b \
  --use-wandb \
  --wandb-project cs336-alignment-ei \
  --wandb-run-name ei-b
```

`EI-C`

```bash
uv run python scripts/run_expert_iteration.py single \
  --profile cloud \
  --train-device cuda:0 \
  --rollout-device cuda:1 \
  --eval-device cuda:1 \
  --ei-batch-size 2048 \
  --rollouts-per-question 8 \
  --num-epochs 2 \
  --n-ei-steps 5 \
  --output-dir /root/autodl-tmp/assignment5-assets/results/expert_iteration/ei_c \
  --use-wandb \
  --wandb-project cs336-alignment-ei \
  --wandb-run-name ei-c
```

如果预算紧张，最少也要完成下面两组，这样才能覆盖至少两种 `G` 和两种 `epoch`：

- `Db=512, G=4, epochs=1`
- `Db=1024 或 2048, G=8, epochs=2`

如果还有预算，优先补 `Db=1024, G=4, epochs=2`，这样可以更干净地区分“增大 rollout”与“增加训练 epoch”各自的贡献。

### 5.3 结果不理想时的扩展顺序

如果 `avg_answer_reward < 15%`，按下面顺序加预算，而不是同时乱调：

1. 先增大 `Db`：`512 -> 1024 -> 2048`
2. 再增大 `G`：`4 -> 8`
3. 最后再增大 step 内 epoch：`1 -> 2`

原因：

- 先增大 `Db`，通常能更稳定提高过滤后的样本量。
- 再增大 `G`，会提高探索，但也显著增加 rollout 成本。
- 最后再增大 epoch，避免在低质量 EI 数据上过拟合。

一个更保守的 2 卡正式命令：

```bash
uv run python scripts/run_expert_iteration.py single \
  --profile cloud \
  --train-device cuda:0 \
  --rollout-device cuda:1 \
  --eval-device cuda:1 \
  --ei-batch-size 512 \
  --rollouts-per-question 4 \
  --num-epochs 1 \
  --n-ei-steps 2 \
  --eval-num-examples 256 \
  --entropy-num-examples 64 \
  --output-dir /root/autodl-tmp/assignment5-assets/results/expert_iteration/cloud_sanity
```

## 6. 每个 EI step 应记录的指标

最低限度建议记录这些字段：

- `ei_step`
- `Db`
- `G`
- `num_rollouts_total`
- `num_correct_rollouts`
- `correct_rollout_ratio`
- `train_num_examples`
- `train_loss`
- `grad_norm`
- `eval_avg_reward`
- `eval_avg_format_reward`
- `eval_avg_answer_reward`
- `mean_token_entropy`
- `mean_response_length`
- `best_checkpoint_dir`

其中最重要的图有两张：

- `ei_step -> eval_avg_answer_reward`
- `train_step 或 ei_step -> mean_token_entropy`

当前脚本里这些指标主要分布在两个地方：

- 顶层 `summary.json`
- 各 step 目录下的 `step_summary.json` 和 `history.json`

查看顶层 summary：

```bash
python -m json.tool cs336_alignment/results/expert_iteration/ei_a/summary.json
```

查看某一步的详细结果：

```bash
python -m json.tool cs336_alignment/results/expert_iteration/ei_a/step_1/step_summary.json
```

## 7. 建议的目录组织

建议把每组 EI 配置单独放一个结果目录，例如：

```text
cs336_alignment/results/expert_iteration/
  ei_a/
    step_1/
      rollout_samples.jsonl
      filtered_sft.jsonl
      history.json
      best_checkpoint/
    step_2/
    ...
    summary.json
  ei_b/
  ei_c/
```

这样后面画图、写报告、回溯某一步失败原因都会容易很多。

## 8. 推荐的执行顺序

### 第 1 天：实现与 smoke

- 跑 `smoke` 命令。
- 检查 `step_1/rollout_samples.jsonl` 是否生成。
- 检查 `step_1/filtered_sft.jsonl` 是否非空。
- 检查 `step_1/history.json`、`step_1/step_summary.json`、顶层 `summary.json` 是否写出。

### 第 2 天：第一组正式 EI

- 跑 `EI-A`。
- 检查 5 个 step 中：
  - 准确率是否持续提升
  - 熵是否明显塌陷
  - 过滤后数据量是否过小

### 第 3 天：第二组与第三组 EI

- 跑 `EI-B`。
- 有预算再跑 `EI-C`。
- 聚合三组配置结果，选最佳模型作为 EI 最终提交模型。

按当前脚本，推荐的实际执行顺序与命令是：

1. Smoke

```bash
uv run python scripts/run_expert_iteration.py single \
  --profile local-smoke \
  --train-device cuda:0 \
  --rollout-device cuda:1 \
  --eval-device none \
  --output-dir /root/autodl-tmp/assignment5-assets/results/expert_iteration/smoke
```

2. EI-A

```bash
uv run python scripts/run_expert_iteration.py single \
  --profile cloud \
  --train-device cuda:0 \
  --rollout-device cuda:1 \
  --eval-device cuda:1 \
  --ei-batch-size 512 \
  --rollouts-per-question 4 \
  --num-epochs 1 \
  --n-ei-steps 5 \
  --output-dir /root/autodl-tmp/assignment5-assets/results/expert_iteration/ei_a
```

3. EI-B

```bash
uv run python scripts/run_expert_iteration.py single \
  --profile cloud \
  --train-device cuda:0 \
  --rollout-device cuda:1 \
  --eval-device cuda:1 \
  --ei-batch-size 1024 \
  --rollouts-per-question 4 \
  --num-epochs 2 \
  --n-ei-steps 5 \
  --output-dir /root/autodl-tmp/assignment5-assets/results/expert_iteration/ei_b
```

4. EI-C

```bash
uv run python scripts/run_expert_iteration.py single \
  --profile cloud \
  --train-device cuda:0 \
  --rollout-device cuda:1 \
  --eval-device cuda:1 \
  --ei-batch-size 2048 \
  --rollouts-per-question 8 \
  --num-epochs 2 \
  --n-ei-steps 5 \
  --output-dir /root/autodl-tmp/assignment5-assets/results/expert_iteration/ei_c
```

5. 汇总查看每组结果

```bash
python -m json.tool cs336_alignment/results/expert_iteration/ei_a/summary.json
python -m json.tool cs336_alignment/results/expert_iteration/ei_b/summary.json
python -m json.tool cs336_alignment/results/expert_iteration/ei_c/summary.json
```

## 9. 报告写作时的结论模板

最终报告至少要能回答下面 4 个问题：

1. EI 相比原始 SFT 有没有提升，提升了多少。
2. 哪组 `Db / G / epochs` 最有效。
3. EI 的提升主要发生在前几个 step，还是后期仍持续增长。
4. 熵曲线是在逐步下降、稳定，还是过早塌陷，这和准确率变化是否一致。

可直接套用的 2 句讨论模板：

- `Compared with our best SFT model, expert iteration improved validation answer reward from X to Y, showing that filtering self-generated correct trajectories can bootstrap better reasoning behavior.`
- `Most of the gain happened in EI steps A-B, while later steps showed diminishing returns / continued improvement, and the entropy curve suggests the model became more confident without immediately collapsing.`

## 10. 一句话版本

这次 EI 的最优策略不是“一上来就大规模扫参”，而是直接使用当前脚本的 `single` 命令，按 `smoke -> EI-A -> EI-B -> EI-C` 的顺序逐步扩实验，确保每一步都留下 `summary.json`、`step_summary.json`、`history.json` 和 checkpoint 这些可画图、可对比、可写报告的结构化结果。
