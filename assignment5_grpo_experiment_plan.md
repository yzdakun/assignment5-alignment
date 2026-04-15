# Assignment 5 第 8 章 GRPO 实验目标与规划

## 1. 总目标与验收标准

基于 [cs336_spring2025_assignment5_alignment_zh.md](cs336_spring2025_assignment5_alignment_zh.md) 第 8 章与当前仓库实现，GRPO 阶段的目标不是只把几个 loss 函数补完，而是完成一条可复现的 `rollout -> reward -> advantage -> policy update -> evaluate` 闭环，并产出可直接写进报告的消融实验结果。

本阶段建议分成两个层面的验收。

### 1.1 训练闭环验收

- 以 `Qwen2.5-Math-1.5B Base` 为起点，在 MATH `train.jsonl` 上完成端到端 GRPO 训练。
- 使用第 7.2 节给出的默认 on-policy 超参数，确认验证奖励随训练提升，且 rollout 质量随时间改善。
- 每 `5` 或 `10` 个 GRPO step 至少在 `1024` 个验证样本上评估一次。
- 记录并保存：
  - `avg_reward`
  - `avg_format_reward`
  - `avg_answer_reward`
  - `grad_norm`
  - `mean_token_entropy`
  - `mean_response_length`
  - `clip_fraction`（仅 off-policy / clip 类损失）
  - 若干 rollout 样例
- 至少产出一个在 MATH validation 上达到 `25%` `avg_answer_reward` 的 GRPO 模型。

### 1.2 第 8 章正式实验验收

按章节要求，最终至少要完成下面 7 组实验或消融：

1. 学习率扫描 `grpo_learning_rate`
2. 基线影响 `grpo_baselines`
3. 长度归一化 `grpo_length_normalization`
4. 分组标准差归一化 `grpo_group_standard_deviation`
5. 离策略 GRPO 扫描 `grpo_off_policy_sweep`
6. Off-policy 下去掉 clipping 的消融 `grpo_off_policy_clip_ablation`
7. prompt 消融 `grpo_prompt_ablation`

每组实验都应至少交付：

- 验证 `avg_answer_reward` 曲线
- 一段简短观察，说明是否出现稳定性、熵、长度或梯度范数上的明显趋势
- 可复现实验配置

## 2. 当前仓库现状与可复用组件

### 2.1 已有能力

- [cs336_alignment/losses.py](cs336_alignment/losses.py)
  - 已有 `masked_mean`
  - 已有 `masked_normalize`
  - 已有 `compute_grpo_clip_loss`
  - 已有 `compute_policy_gradient_loss`
  - 已有 `grpo_microbatch_train_step`
- [cs336_alignment/rewards.py](cs336_alignment/rewards.py)
  - 已有 `compute_group_normalized_rewards`
  - 已导出 `r1_zero_reward_fn`
  - 已导出 `question_only_reward_fn`
- [cs336_alignment/logging_utils.py](cs336_alignment/logging_utils.py)
  - 已能统计 `mean_token_entropy`
  - 已能统计 `mean_response_length`
  - 已能保存带奖励与熵的生成样例
- [cs336_alignment/sft.py](cs336_alignment/sft.py)
  - 已有成熟的训练主循环
  - 已有 `history.json`、`best_checkpoint`、`final_checkpoint` 的保存方式
  - 已有 vLLM 验证评测和 wandb 记录范式
- [cs336_alignment/expert_iteration.py](cs336_alignment/expert_iteration.py)
  - 已有双 GPU 下“训练模型 + rollout/eval vLLM”的组织方式
  - 已有分步 summary、熵统计、验证集评测和输出目录组织方式
- [scripts/run_grpo.py](scripts/run_grpo.py)
  - 已有统一入口脚本，可作为后续 GRPO CLI 的调用入口

### 2.2 当前缺口

- [cs336_alignment/grpo.py](cs336_alignment/grpo.py) 现已补齐单次训练、sweep 与章节 8 聚合运行 CLI，但仍需要通过正式 cloud 训练验证各命令在真实双 GPU 环境下的稳定性。
- 当前仓库已经有 GRPO 专属结果目录约定、summary 结构和 sweep 脚本，但还需要根据磁盘预算决定哪些实验保留 checkpoint，哪些实验只保留 `history.json` / `summary.json`。
- `tests/test_grpo.py` 虽然存在，但本地直接运行 `uv run pytest tests/test_grpo.py -q` 时会先卡在 `flash-attn` 构建，当前环境缺少 `nvcc` / `CUDA_HOME`，因此还不能把“测试已通过”作为现状结论。

### 2.3 当前可用基线

- [cs336_alignment/results/math_baseline/summary.json](cs336_alignment/results/math_baseline/summary.json) 显示 base 模型当前 `avg_answer_reward = 0.0226`，这是最直接的 GRPO 起跑线。
- [results/EI/summary/step_summary.json](results/EI/summary/step_summary.json) 中已有 EI 最终 `final_eval avg_answer_reward = 0.351`，说明当前模型、数据与评测链路是有希望超过第 8 章 `25%` 目标的。
- [cs336_alignment/results/sft_experiment/history.json](cs336_alignment/results/sft_experiment/history.json) 目前是 local smoke 级别的 SFT 记录，不适合直接当成正式 GRPO 对照实验。

结论：当前仓库并不是“从零开始做 GRPO”，而是“GRPO 的核心算子和评测部件大多已具备，缺的是完整 trainer、实验编排和结果沉淀”。

## 3. 推荐实现路径

### 阶段 A：先解决环境前置问题

目标：确保后续最基本的 GRPO 单测和训练命令可运行。

建议先检查：

```bash
nvcc --version
echo $CUDA_HOME
uv run pytest -k test_compute_group_normalized_rewards
uv run pytest -k test_compute_grpo_clip_loss
uv run pytest -k test_grpo_microbatch_train_step
```

如果 `flash-attn` 仍无法构建，优先解决环境，而不是直接开始长时间实验。否则后续很容易在正式跑 GRPO 时中途失败。

### 阶段 B：补齐最小可运行 GRPO 闭环

目标：先让 [cs336_alignment/grpo.py](cs336_alignment/grpo.py) 能稳定跑完一个短程 on-policy 训练。

建议直接复用 [cs336_alignment/sft.py](cs336_alignment/sft.py) 和 [cs336_alignment/expert_iteration.py](cs336_alignment/expert_iteration.py) 的结构，把 GRPO 主循环拆成下面几步：

1. 从训练集中抽取 `n_prompts_per_rollout_batch = rollout_batch_size // group_size` 个问题。
2. 用 `r1_zero.prompt` 构造 prompt，并用 vLLM 采样 `group_size` 条响应。
3. 用 `compute_group_normalized_rewards` 计算：
   - `raw_rewards`
   - `advantages`
   - reward 统计信息
4. 将 prompt 和 response 重新 token 化，得到：
   - `input_ids`
   - `labels`
   - `response_mask`
5. 用 `get_response_log_probs` 计算当前策略对 rollout 的 token logprob。
6. 调用 `grpo_microbatch_train_step` 做一次或多次梯度更新。
7. 每隔若干 step 用 vLLM 做验证评测，并保存：
   - `history.json`
   - `summary.json`
   - `best_checkpoint`
   - `final_checkpoint`

### 阶段 C：先跑一个 smoke 级 on-policy GRPO

目标：验证数据流、梯度流和日志流正确，不追求分数。

建议 smoke 配置：

- `n_grpo_steps = 10 ~ 20`
- `rollout_batch_size = 64`
- `group_size = 4`
- `epochs_per_rollout_batch = 1`
- `train_batch_size = 64`
- `gradient_accumulation_steps` 调到显存能接受即可
- `loss_type = "reinforce_with_baseline"`
- `use_std_normalization = True`
- `eval_num_examples = 128`

smoke 成功标准：

- 无空响应、无 NaN、无 reward shape 错误
- `avg_answer_reward` 至少高于 base 附近的随机波动
- rollout 样例开始出现合理的 `<think> ... </think> <answer> ... </answer>` 结构
- `history.json`、checkpoint、评测结果都能落盘

推荐命令：

```bash
.venv/bin/python scripts/run_grpo.py single \
  --profile local-smoke \
  --output-dir results/GRPO/smoke \
  --save-rollouts \
  --skip-save-checkpoint \
  --skip-save-best-checkpoint
```

### 阶段 D：跑第 7.2 节默认 on-policy 主线

在 smoke 正常后，建议先严格对齐文档默认配置，做第一条正式 GRPO 主线：

- `n_grpo_steps = 200`
- `learning_rate = 1e-5`
- `advantage_eps = 1e-6`
- `rollout_batch_size = 256`
- `group_size = 8`
- `sampling_temperature = 1.0`
- `sampling_min_tokens = 4`
- `sampling_max_tokens = 1024`
- `epochs_per_rollout_batch = 1`
- `train_batch_size = 256`
- `gradient_accumulation_steps = 128`
- `gpu_memory_utilization = 0.85`
- `loss_type = "reinforce_with_baseline"`
- `use_std_normalization = True`
- `optimizer = AdamW(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95))`
- `max_grad_norm = 1.0`

这条主线的意义是：

- 先确认“按讲义默认值”时，仓库实现是否能出现合理提升
- 为后续第 8 章所有消融提供统一起点

推荐命令：

```bash
.venv/bin/python scripts/run_grpo.py single \
  --profile cloud \
  --output-dir results/GRPO/on_policy_main
```

## 4. 第 8 章实验矩阵建议

### 4.1 学习率扫描

目标：找到第一条能稳定超过 `25%` 验证准确率的主线配置。

建议从默认 `1e-5` 向两侧扫描，优先尝试：

- `3e-6`
- `1e-5`
- `3e-5`
- 如前 3 个都稳定，可再补 `1e-4` 作为发散边界

建议执行方式：

- 每组先跑到 `50` step 看趋势
- 明显发散或明显次优时提前停止
- 只把最有希望的 `1 ~ 2` 组继续跑满 `200` step

推荐命令：

```bash
.venv/bin/python scripts/run_grpo.py learning-rate \
  --profile cloud \
  --output-dir results/GRPO/learning_rate
```

如果磁盘紧张，建议在学习率扫描阶段关闭 checkpoint 保存，只保留结果文件。

### 4.2 基线影响

在最佳学习率上比较：

- `no_baseline`
- `reinforce_with_baseline`

预期：

- `reinforce_with_baseline` 大概率更稳
- `no_baseline` 可能 reward 方差更大，梯度和验证曲线更抖

推荐命令：

```bash
.venv/bin/python scripts/run_grpo.py baselines \
  --profile cloud \
  --output-dir results/GRPO/baselines
```

这组实验通常只需要保存曲线与 summary，不需要保留两条权重。

### 4.3 长度归一化

比较：

- `masked_mean`
- `masked_normalize`

建议：

- 先把默认主线实现成 `masked_mean`
- 再把训练聚合改成 `masked_normalize`
- `normalize_constant` 建议先固定为 `sampling_max_tokens`，保持不同响应长度之间的 credit assignment 可比较

重点观察：

- `grad_norm`
- `mean_response_length`
- `avg_answer_reward`

推荐命令：

```bash
.venv/bin/python scripts/run_grpo.py length-normalization \
  --profile cloud \
  --output-dir results/GRPO/length_normalization
```

### 4.4 分组标准差归一化

比较：

- `use_std_normalization = True`
- `use_std_normalization = False`

重点观察：

- 是否有更稳的奖励提升
- 是否减少“简单题/困难题”权重异常
- 梯度范数和奖励方差是否更平滑

推荐命令：

```bash
.venv/bin/python scripts/run_grpo.py std-normalization \
  --profile cloud \
  --output-dir results/GRPO/std_normalization
```

### 4.5 Off-policy GRPO 扫描

这是第 8 章最值得投入时间的一组实验。

固定：

- `rollout_batch_size = 256`
- `loss_type = "grpo_clip"`

建议先做一个少于 `50` step 的广泛扫描，再做一个 `200` step 的集中扫描。

广泛扫描建议优先覆盖下面几组：

| 配置名 | `epochs_per_rollout_batch` | `train_batch_size` | 用途 |
| --- | ---: | ---: | --- |
| OP-0 | 1 | 256 | on-policy 参考线 |
| OP-1 | 2 | 256 | 轻度 off-policy |
| OP-2 | 4 | 256 | 更强重复利用 rollout |
| OP-3 | 2 | 128 | 更小 train batch，增加 optimizer updates |
| OP-4 | 4 | 128 | 更激进的 off-policy 对比 |

注意事项：

- 必须同步调整 `gradient_accumulation_steps`，保持显存占用不失控
- `old_log_probs` 只在每个 rollout batch 开始时算一次，并在所有 epoch 中复用
- `old_log_probs` 必须放在 `torch.inference_mode()` 下，且不能参与反传

集中扫描时，保留广泛扫描里最好的 `2 ~ 3` 组，跑满 `200` step，并与 on-policy 主线同时比较：

- 相对于 validation step 的提升速度
- 相对于 wall-clock time 的提升速度

推荐命令：

```bash
.venv/bin/python scripts/run_grpo.py off-policy-sweep \
  --profile cloud \
  --output-dir results/GRPO/off_policy
```

如果想自定义扫描网格：

```bash
.venv/bin/python scripts/run_grpo.py off-policy-sweep \
  --profile cloud \
  --configs 1x256,2x256,4x256,2x128,4x128 \
  --output-dir results/GRPO/off_policy
```

这组实验建议只保留表现最好的 `grpo_clip` 权重，其余 run 只保留结果。

### 4.6 Off-policy No-Clip 消融

在最佳 off-policy 配置上，对比：

- `grpo_clip`
- `grpo_no_clip`

重点看：

- 验证奖励是否更快上涨但更不稳定
- `clip_fraction` 消失后，`grad_norm` 是否明显变大
- 熵与输出长度是否更容易异常漂移

推荐命令：

```bash
.venv/bin/python scripts/run_grpo.py off-policy-no-clip \
  --profile cloud \
  --output-dir results/GRPO/off_policy_no_clip
```

这组实验通常只需要保留结果，不需要额外保存 `grpo_no_clip` 权重。

### 4.7 Prompt 消融

比较：

- `r1_zero.prompt` + `r1_zero_reward_fn`
- `question_only.prompt` + `question_only_reward_fn`

建议：

- 训练和验证必须同时切 prompt 和 reward_fn
- 其他超参数固定为当前最好配置

重点看：

- 格式奖励是否显著变化
- 平均响应长度是否明显变短
- 正确率变化是否主要来自“更短更直接”还是“思维链减少导致推理退化”

推荐命令：

```bash
.venv/bin/python scripts/run_grpo.py prompt-ablation \
  --profile cloud \
  --output-dir results/GRPO/prompt_ablation
```

如果其中某条 prompt 配置最终成为你的最优模型，再单独保留那条权重即可。

### 4.8 一键跑完第 8 章要求

如果你希望按当前脚本封装，把章节 8 要求的实验顺序整体串起来跑，可以使用聚合命令：

```bash
.venv/bin/python scripts/run_grpo.py all-required \
  --profile cloud \
  --output-dir results/GRPO/chapter8_required
```

这个命令会依次执行：

1. `learning-rate`
2. `baselines`
3. `length-normalization`
4. `std-normalization`
5. `off-policy-sweep`
6. `off-policy-no-clip`
7. `prompt-ablation`

并在每个阶段自动读取上一阶段选出的最佳配置。

## 5. 结果产物与目录建议

建议仿照 EI 和 SFT 的产物组织方式，新增：

```text
results/GRPO/
  smoke/
  lr_scan/
  baselines/
  length_norm/
  std_norm/
  off_policy/
  off_policy_no_clip/
  prompt_ablation/
```

每个实验子目录至少包含：

- `config.json`
- `history.json`
- `summary.json`
- `best_checkpoint/`
- `final_checkpoint/`
- `samples/` 或 `rollouts/`
- `fig/`

在磁盘紧张时，可以把产物目标分成两档：

- 必留：
  - `history.json`
  - `summary.json`
  - 画图和写报告所需样例
- 选留：
  - `best_checkpoint/`
  - `final_checkpoint/`

推荐策略：

- `single` 正式主线：建议保留至少一份最好权重
- `learning-rate` / `baselines` / `length-normalization` / `std-normalization`：通常只保留结果
- `off-policy-sweep`：只保留最佳 `grpo_clip` 权重
- `off-policy-no-clip`：通常只保留结果
- `prompt-ablation`：只有当某条 prompt 最终成为最佳方案时才保留权重

建议 `history.json` 中统一记录：

- `train_step`
- `eval_step`
- `avg_reward`
- `avg_format_reward`
- `avg_answer_reward`
- `loss`
- `grad_norm`
- `mean_token_entropy`
- `mean_response_length`
- `clip_fraction`
- `wall_clock_seconds`

这样后续画图和写报告会非常顺手。

## 6. 风险点与规避建议

### 6.1 环境风险

当前最现实的问题不是算法，而是环境：

- `flash-attn` 构建依赖 `nvcc`
- 需要正确设置 `CUDA_HOME`
- 如果这一层没处理好，单测和正式训练都会被阻塞

### 6.2 Trainer 还没落地

当前仓库已经有 [cs336_alignment/grpo.py](cs336_alignment/grpo.py) 与 [scripts/run_grpo.py](scripts/run_grpo.py) 的训练和实验 CLI，但最主要的工程风险已经从“没有 trainer”变成了“真实 cloud 环境下是否稳定、是否省磁盘、是否能在预算内跑完”。因此建议不要一上来就做 sweep，而是先做：

1. smoke
2. 默认 on-policy 主线
3. 第 8 章消融

### 6.3 评测噪声

CoT / RL 的验证噪声通常较大，因此：

- 调参阶段至少用 `1024` 验证样本
- 最终结论尽量使用完整 `5000` 验证样本
- 不要只根据 `32` 或 `64` 个样本下结论

### 6.4 Off-policy 不稳定

off-policy 会显著提高 sample efficiency，但也更容易出问题。建议重点监控：

- `clip_fraction`
- `grad_norm`
- `mean_token_entropy`
- `mean_response_length`

如果出现：

- 熵快速塌陷
- 长度异常缩短或爆长
- 梯度范数持续过大
- 验证奖励突然掉头

应优先回退到更保守的 `epochs_per_rollout_batch` 或更大的 `train_batch_size`。

## 7. 推荐执行顺序

为了减少无效 GPU 消耗，建议按下面顺序推进：

1. 修好环境，确保 GRPO 相关单测能跑。
2. 实现并跑通 `grpo.py` 的 smoke 版本。
3. 跑默认 on-policy `200` step 主线，确认奖励确实上升。
4. 做学习率扫描，锁定一组主学习率。
5. 做 `baseline`、`length normalization`、`std normalization` 三个 on-policy 消融。
6. 实现并扫描 off-policy GRPO-Clip。
7. 在最佳 off-policy 配置上做 No-Clip 消融。
8. 最后做 prompt 消融。

对应脚本顺序为：

```bash
.venv/bin/python scripts/run_grpo.py single --profile local-smoke --output-dir results/GRPO/smoke --save-rollouts --skip-save-checkpoint --skip-save-best-checkpoint
.venv/bin/python scripts/run_grpo.py single --profile cloud --output-dir results/GRPO/on_policy_main
.venv/bin/python scripts/run_grpo.py learning-rate --profile cloud --output-dir results/GRPO/learning_rate
.venv/bin/python scripts/run_grpo.py baselines --profile cloud --output-dir results/GRPO/baselines
.venv/bin/python scripts/run_grpo.py length-normalization --profile cloud --output-dir results/GRPO/length_normalization
.venv/bin/python scripts/run_grpo.py std-normalization --profile cloud --output-dir results/GRPO/std_normalization
.venv/bin/python scripts/run_grpo.py off-policy-sweep --profile cloud --output-dir results/GRPO/off_policy
.venv/bin/python scripts/run_grpo.py off-policy-no-clip --profile cloud --output-dir results/GRPO/off_policy_no_clip
.venv/bin/python scripts/run_grpo.py prompt-ablation --profile cloud --output-dir results/GRPO/prompt_ablation
```

### 7.1 实验执行总表

下面这张表把每个任务的执行重点汇总到一起。完整命令保留在上面的第 4 节和第 7 节命令清单里，这里主要用于排期、控盘和做实验记录。

说明：

- 第 8 章给出的 `6 / 2 / 2 / 2 / 12 / 2 / 2 H100 hours` 是文档原始预算。
- `smoke` 和 `默认 on-policy 主线` 的耗时是经验估计，不是作业文档硬性数字。
- “是否保存权重”指建议，而不是强制要求。

| 任务 | 脚本命令 | 建议步数 | 预计 GPU 时间 | 是否建议保存权重 | 建议输出目录 |
| --- | --- | ---: | --- | --- | --- |
| Smoke 闭环验证 | `single --profile local-smoke` | `10~20` | `10~30` 分钟 | 否 | `results/GRPO/smoke` |
| 默认 on-policy 主线 | `single --profile cloud` | `200` | 约 `1~2` H100 小时 | 是，至少保留 1 份最佳权重 | `results/GRPO/on_policy_main` |
| 学习率扫描 | `learning-rate --profile cloud` | 先 `<50`，再 `200` | `6 H100 hours` | 通常否，只保留最佳一条即可 | `results/GRPO/learning_rate` |
| 基线影响 | `baselines --profile cloud` | `200` | `2 H100 hours` | 否，通常只保留结果 | `results/GRPO/baselines` |
| 长度归一化 | `length-normalization --profile cloud` | `200` | `2 H100 hours` | 否，通常只保留结果 | `results/GRPO/length_normalization` |
| 分组标准差归一化 | `std-normalization --profile cloud` | `200` | `2 H100 hours` | 否，通常只保留结果 | `results/GRPO/std_normalization` |
| Off-policy 广泛扫描 | `off-policy-sweep --profile cloud` | `<50` | 包含在 `12 H100 hours` 内 | 否，通常只保留结果 | `results/GRPO/off_policy` |
| Off-policy 集中扫描 | `off-policy-sweep --profile cloud` | `200` | 包含在 `12 H100 hours` 内 | 是，只保留最佳 `grpo_clip` | `results/GRPO/off_policy` |
| Off-policy No-Clip 消融 | `off-policy-no-clip --profile cloud` | `200` | `2 H100 hours` | 否，通常只保留结果 | `results/GRPO/off_policy_no_clip` |
| Prompt 消融 | `prompt-ablation --profile cloud` | `200` | `2 H100 hours` | 否，除非某条 prompt 成为最终最佳方案 | `results/GRPO/prompt_ablation` |
| 第 8 章一键执行 | `all-required --profile cloud` | 自动串联 | 取决于所有子实验总和 | 否，除非你手动保留阶段最佳模型 | `results/GRPO/chapter8_required` |

### 7.2 最省空间的执行策略

如果数据盘非常紧张，建议采用下面的执行策略：

1. `smoke`、`learning-rate`、`baselines`、`length-normalization`、`std-normalization`、`off-policy-no-clip`、`prompt-ablation`
   只保留 `history.json`、`summary.json`、必要样例，不保留 checkpoint。
2. `默认 on-policy 主线`
   保留 1 份最佳权重，作为 on-policy 参考线。
3. `off-policy-sweep`
   只保留最终最好的 `grpo_clip` 权重，其余 run 只保留结果。
4. `最终最优模型`
   单独复制或保留到一个明确目录，避免被后续实验覆盖。

如果你后面真的按这个策略执行，最关键需要保留的通常只有两类模型：

- `best_on_policy_checkpoint`
- `best_overall_checkpoint`

## 8. 一句话结论

这份规划的核心判断是：当前仓库已经具备完成第 8 章 GRPO 实验所需的 trainer 与 CLI 封装，接下来最稳妥的路径是先跑通 smoke 和默认 on-policy 参考线，再按“学习率 -> 三个 on-policy 消融 -> off-policy -> No-Clip -> prompt” 的顺序逐步扩展实验，并根据磁盘预算只保留真正需要复用的权重。
