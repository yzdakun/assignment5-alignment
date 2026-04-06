# Assignment 5 与 Open-R1 的工业工程流水线对比学习笔记

## 文档目标

这份笔记的目标不是重复解释 Assignment 5 的算法定义，而是回答一个更偏工程的问题：

- 如果我按照 Assignment 5 去做，我学到的是哪些“算法原语”？
- 如果我想进一步理解工业界或开源社区会怎样把这些任务落成一条训练流水线，应该看什么？
- 参考 [Open-R1](https://github.com/huggingface/open-r1) 时，Assignment 5 里的每个任务在工业实现上通常会变成什么样子？

可以把两者先粗略地理解为：

- `Assignment 5`：教学版、手写版、受控实验版的 reasoning post-training。
- `Open-R1`：框架版、工程化版、面向公开复现与扩展的 R1 训练系统。

Assignment 5 的核心价值在于让你亲手实现 `mask / logprob / advantage / clipping / on-off policy` 这些关键原语；Open-R1 的核心价值在于展示工业界或开源研究工程里，如何用成熟框架把这些能力组织成稳定、可扩展、可复现的训练系统。

## 一页总览

| Assignment 5 任务 | 作业中的重点 | 工业界常见实现方式 | Open-R1 中的对应方式 | 对比学习重点 |
|---|---|---|---|---|
| Zero-shot baseline | 自己写评测脚本，理解 prompt、采样和 reward | 用高吞吐推理引擎 + 标准化 benchmark/eval 框架 | `vLLM` + `lighteval` + 统一评测命令 | 从“写评测函数”升级到“设计评测服务” |
| SFT 辅助方法 | 手写 tokenization、response mask、logprob、entropy | 交给 `Trainer`/数据管线，自己关注模板、数据、超参 | `trl.SFTTrainer` + `Accelerate/DeepSpeed` | 理解原语后，再学会配置与调度框架 |
| SFT 实验 | 用固定数学数据做监督微调并比较数据量/过滤策略 | 数据版本管理、蒸馏数据生成、过滤、混合、长上下文训练 | `Mixture-of-Thoughts` / `OpenR1-Math-220k` + YAML recipes | 训练脚本只是末端，数据工程才是上游杠杆 |
| Expert Iteration | 采样自己的推理轨迹，过滤正确样本，再继续 SFT | 一般做成数据生成 + 验证 + 过滤 + 再训练的离线闭环 | `generate.py` + pass-rate filtering + 数据集发布 | 工业里 EI 常被“数据闭环”吸收，而不是单独起名 |
| GRPO 原语 | 手写 group normalization、clip loss、masked mean | 使用成熟 RL trainer，重点转向 reward 设计、稳定性与吞吐 | `trl.GRPOTrainer` + reward registry + `vLLM` backend | 从“实现 GRPO”过渡到“运营 GRPO 系统” |
| GRPO 训练循环 | 双卡手写 rollout/train loop，记录 entropy、reward 等指标 | 多卡多节点、ZeRO、容错恢复、日志平台、配置驱动 | `Accelerate` / `DeepSpeed` / Slurm / W&B | 训练循环从 Python for-loop 变成分布式 job |
| 消融实验 | 学习率、baseline、长度归一化、prompt 等系统性对比 | sweep 平台、标准 recipe、多维指标 dashboard | YAML 配置、callbacks、benchmark 脚本 | 工业里 ablation 的关键是实验组织能力 |
| 最终优化/排行榜 | 在固定 2 卡预算下尽量做高分 | 系统吞吐优化、任务并行、服务化 rollout、模型/数据发布 | ZeRO, TP/DP, `vllm_mode`, Hub 发布 | 算法效果和系统效率是同一个问题的两面 |

## 1. Zero-shot baseline：从“评测脚本”到“评测流水线”

### Assignment 5 在做什么

作业要求你：

- 在 `MATH` 验证集上建立 zero-shot baseline。
- 使用固定的 `r1_zero` prompt。
- 用 `vLLM` 批量采样。
- 用答案解析器计算格式奖励和答案奖励。
- 将样本、模型输出和分数序列化保存，便于后续分析。

这一步的重点是让你理解：推理任务的评测不是简单看交叉熵，而是要把“生成 + 解析 + 验证”串起来。

### 工业界通常怎么做

工业界或成熟开源项目不会把 zero-shot baseline 当成一个一次性脚本，而会把它做成一条评测流水线：

- 用 `vLLM` 或 `SGLang` 作为统一推理后端，支持高吞吐批量生成。
- 用标准 benchmark 框架，例如 `lighteval`，统一管理任务、采样参数、结果格式与对比表。
- 统一处理 prompt template、chat template、EOS 对齐和 stop strings。
- 将推理结果、采样参数、模型 revision、benchmark 版本一起记录到 artifact 系统中。
- 小 benchmark 常用 `pass@k` 或多次采样估计，而不是只看单次 greedy/单次 sample。

### Open-R1 是怎么做的

Open-R1 将评测独立成一层标准流程：

- 使用 `lighteval` 做多 benchmark 评测，例如 `AIME 2024`、`MATH-500`、`GPQA Diamond`、`LiveCodeBench`。
- 评测入口统一走 `vLLM`，而不是每个实验自己写一套 sampling 逻辑。
- 支持单卡、多卡数据并行和 tensor parallel 的评测方式。
- 在 README 中明确给出采样参数和复现 DeepSeek 报告结果的方法。

参考：

- [Open-R1 README: Evaluating models](https://github.com/huggingface/open-r1/blob/main/README.md)

### 你应该对比学习什么

- Assignment 5 教你的是：`reward_fn` 背后到底在判定什么。
- Open-R1 教你的是：评测系统如何标准化、如何支持多模型、多 benchmark、多 GPU。

建议把这一步理解成：

- 作业版本：我会写“评测逻辑”
- 工业版本：我会搭“评测平台”

## 2. SFT 辅助原语：从手写张量操作到 Trainer 抽象

### Assignment 5 在做什么

作业显式要求你手写一批 SFT/RL 通用原语：

- `tokenize_prompt_and_output`
- `compute_entropy`
- `get_response_log_probs`
- `masked_normalize`
- `sft_microbatch_train_step`
- `log_generations`

这部分是整份作业最“底层”的一段。它逼着你弄清楚：

- 为什么只在 response token 上算 loss。
- `input_ids` 和 `labels` 是怎么移位的。
- token-level logprob 是怎么从 logits 里取出来的。
- entropy 为什么能帮助你判断模型是否过度自信。
- gradient accumulation 在 microbatch 级别怎么体现。

### 工业界通常怎么做

工业界一般不会在主工程里重复手写这些原语，而是交给训练框架：

- `trl.SFTTrainer` 或内部 SFT trainer 负责 loss、batch、padding、backward。
- `Accelerate` / `DeepSpeed` 负责分布式、ZeRO、梯度累积和恢复训练。
- 数据预处理管线负责把原始样本转成对话格式或指令格式。
- 重点从“如何写一个 cross-entropy loop”转移到：
  - 聊天模板是否正确；
  - EOS token 是否和模板一致；
  - 上下文长度是否足够；
  - 是否用 gradient checkpointing；
  - 是否使用 FlashAttention、Liger 等加速组件；
  - 是否有断点恢复、artifact 存储和模型发布。

### Open-R1 是怎么做的

Open-R1 的 `sft.py` 几乎没有重写 SFT 的数学原语，而是：

- 用 `trl.SFTTrainer` 完成训练。
- 用 `TrlParser` 和 YAML 配置组织参数。
- 用 `Accelerate` 启动训练。
- 支持 `DeepSpeed ZeRO-3`。
- 使用 `bf16`、`flash_attention_2`、`gradient_checkpointing`、`use_liger_kernel`。
- 处理 chat template 和 EOS token 对齐问题。
- 自动支持 checkpoint resume、W&B、push to hub。

参考：

- [Open-R1 `sft.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/sft.py)
- [Open-R1 SFT recipe](https://github.com/huggingface/open-r1/blob/main/recipes/OpenR1-Distill-7B/sft/config_distill.yaml)

### 你应该对比学习什么

- Assignment 5 教你“Trainer 帮你藏起来的东西到底是什么”。
- Open-R1 教你“工业里为什么必须把这些细节抽象掉，否则系统不可维护”。

最好的学习顺序是：

1. 先通过作业搞懂原语。
2. 再看 Open-R1 认识这些原语在框架里是怎样被封装的。
3. 最后反推：哪些地方仍然需要人工介入，例如 chat template、数据列映射、日志、恢复训练。

## 3. SFT 实验：从小规模监督微调到数据工程驱动的 SFT

### Assignment 5 在做什么

Assignment 5 的 SFT 实验要求非常清晰：

- 用 `sft.jsonl` 在 `Qwen2.5-Math-1.5B Base` 上做 reasoning SFT。
- 对比不同数据规模。
- 过滤只产生正确答案的样本，再训练一遍。
- 看验证准确率曲线怎么变。

这一步的教学重点是：

- 让你看到数据量和数据质量对 SFT 的影响。
- 让你理解“reasoning trace 不是越多越好，错误轨迹会污染监督信号”。

### 工业界通常怎么做

工业界在 SFT 阶段真正投入最多资源的往往不是训练 loop，而是数据工程：

- 蒸馏更强模型的 reasoning traces。
- 用 verifier 或 pass-rate 对轨迹做过滤。
- 做 dataset mixture、难度分层、长度分桶和去污染。
- 在有限算力下选择全参训练、LoRA 或 QLoRA。
- 用长上下文配方支持更长 CoT。
- 把数据、模板、模型、超参版本化，确保多轮迭代可比。

### Open-R1 是怎么做的

Open-R1 的核心不是“给你一个静态 SFT 文件”，而是提供一整套围绕蒸馏数据的工程基础设施：

- 发布蒸馏得到的 reasoning 数据集，例如 `Mixture-of-Thoughts`、`OpenR1-Math-220k`。
- 提供 `generate.py` 通过 `distilabel + vLLM/OpenAI-compatible server` 大规模生成样本。
- 支持 dataset mixture。
- 提供数据去污染脚本。
- 通过 YAML recipe 标准化 SFT 训练。

参考：

- [Open-R1 README: Data generation](https://github.com/huggingface/open-r1/blob/main/README.md)
- [Open-R1 `generate.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/generate.py)
- [Open-R1 `configs.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/configs.py)

### 你应该对比学习什么

Assignment 5 的“过滤正确样本再训练”对应到工业界，本质上就是：

- 更强的数据清洗
- 更系统的 rejection sampling
- 更长的离线数据闭环

也就是说，作业里的简单过滤，其实是工业数据工程的一个缩小版。

## 4. Expert Iteration：作业里的显式算法，工业里的隐式数据闭环

### Assignment 5 在做什么

Assignment 5 专门把 `Expert Iteration` 拿出来作为一个独立阶段：

- 当前模型对训练集问题采样多条 reasoning trace。
- 只保留能得到正确答案的轨迹。
- 用这些“专家轨迹”继续训练模型。
- 比较不同 rollout 数量、epoch 数量和 EI step 的表现。

它想教你的核心是：

- reasoning 能力不一定完全依赖外部标注数据；
- 模型可以通过“自己生成 -> verifier 过滤 -> 再训练”的方式自举。

### 工业界通常怎么做

工业界并不一定会把这件事显式命名为 `Expert Iteration`，但思路非常常见。常见实现形式是：

- 离线 rejection sampling pipeline
- self-improvement / self-training pipeline
- generate-filter-train 闭环
- pass-rate filtering / verifier filtering

工程上通常拆成四段：

1. 生成 job：大批量从当前模型采样候选答案。
2. 验证 job：用 parser / unit tests / sandbox 执行器给候选答案打分。
3. 数据整理 job：过滤、去重、分桶、合并配置分片。
4. 再训练 job：把过滤后的数据回灌到 SFT 或下一阶段 RL。

### Open-R1 是怎么做的

Open-R1 虽然没有把 `Expert Iteration` 作为一个单独 trainer 暴露出来，但它实际上提供了这个闭环的大部分工程积木：

- `generate.py`：面向大规模数据生成的 pipeline。
- `scripts/pass_rate_filtering`：对可验证任务进行 pass-rate 过滤。
- 数据集分片与重新合并流程。
- 数据发布到 Hub，供下一轮训练复用。

参考：

- [Open-R1 `generate.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/generate.py)
- [Open-R1 pass-rate filtering README](https://github.com/huggingface/open-r1/blob/main/scripts/pass_rate_filtering/README.md)

### 你应该对比学习什么

这部分是 Assignment 5 和工业实现最值得对应起来看的地方：

- 作业版本：显式研究 `EI` 算法本身。
- 工业版本：把 `EI` 吸收到“数据生成与过滤流水线”里。

所以如果你要把作业迁移到工业语境，最自然的映射不是“再写一个 EI trainer”，而是：

- 把 EI 看成一条离线数据闭环。

## 5. GRPO 原语：从“我会推导损失”到“我会运营 RL 训练系统”

### Assignment 5 在做什么

Assignment 5 要你自己实现 GRPO 的核心数学组件：

- `compute_group_normalized_rewards`
- `compute_naive_policy_gradient_loss`
- `compute_grpo_clip_loss`
- `compute_policy_gradient_loss`
- `masked_mean`
- `grpo_microbatch_train_step`

同时它还要求你做一系列算法消融：

- baseline vs no baseline
- length normalization
- std normalization
- on-policy vs off-policy
- clipping ablation

这是非常典型的教学设计：先让你“看见” RL 目标，然后再讨论这些设计选择为什么影响稳定性和性能。

### 工业界通常怎么做

工业界做 RL post-training 时，通常不会在主项目里手写这些损失，而是：

- 用一个成熟 trainer，例如 `TRL GRPOTrainer` 或内部 PPO/GRPO trainer。
- 将 reward 函数做成注册表式、可组合、可加权的模块。
- 关注 rollout throughput、reward latency、稳定性、checkpoint 与恢复。
- 对研究型问题做配置层面的切换，而不是改训练主循环源码。

工业界更常见的关注点是：

- reward 是否鲁棒；
- reward 是否可并行计算；
- rollout 和 reward 评估是否能异步化；
- 训练成本是否可控；
- 老策略 logprob 是否缓存；
- vLLM 与训练进程如何协同。

### Open-R1 是怎么做的

Open-R1 的 `grpo.py` 直接使用 `trl.GRPOTrainer`，没有重写 GRPO 的核心张量逻辑。它把工程重心放在：

- 数据集列映射与 prompt 格式化；
- reward 函数注册与组合；
- tokenizer / model / callbacks / W&B 初始化；
- checkpoint resume；
- hub 发布；
- 和 `vLLM` 后端协作。

同时，Open-R1 的 reward 系统是组合式的，不只包含 `accuracy`，还有：

- `format`
- `tag_count`
- `reasoning_steps`
- `length`
- `cosine`
- `repetition_penalty`
- `code` 相关 reward
- `soft_overlong_punishment`

参考：

- [Open-R1 `grpo.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py)
- [Open-R1 `rewards.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py)

### 你应该对比学习什么

这一步最关键的认知迁移是：

- Assignment 5 在教你“GRPO 的数学骨架”。
- Open-R1 在教你“GRPO 的工程器官”。

如果你只看 Open-R1，容易不知道 `clip loss`、`group normalization`、`old_log_probs` 在背后如何运作。
如果你只做 Assignment 5，又容易不知道工业里为什么必须把这些实现折叠进 trainer，并把注意力转向 reward、模板、并行和容错。

## 6. GRPO 训练循环：从双卡 for-loop 到多节点训练作业

### Assignment 5 在做什么

作业里，GRPO 训练循环被组织成一个清晰的小型系统：

- 1 张 GPU 跑 policy。
- 1 张 GPU 跑 `vLLM`。
- 采样 rollout。
- 计算 reward / advantage。
- 取旧策略 logprob。
- 按 microbatch 回传梯度。
- 记录 loss、reward、entropy、grad norm、clip fraction。

它是一个非常适合教学和调试的“双卡受控系统”。

### 工业界通常怎么做

工业界会把这套流程进一步拆成更显式的系统层：

- 训练层：`Accelerate` / `DeepSpeed ZeRO-2/3` / FSDP。
- rollout 层：`vLLM` colocate 或独立 server。
- 调度层：Slurm / Kubernetes / Ray / 内部平台。
- 观察层：W&B、Prometheus、structured logs、artifact store。
- 数据层：训练集、验证集、生成集、过滤集、benchmark 结果集全部版本化。
- 恢复机制：自动 resume、失败重试、checkpoint 清理策略。

在多节点场景下，工业界通常会：

- 用 1 个或多个专门节点跑 vLLM server。
- 用 N 个节点跑训练。
- 在 rollout 和训练之间保持明确的通信协议与资源隔离。

### Open-R1 是怎么做的

Open-R1 README 里已经是一个典型的研究工程训练栈：

- `accelerate launch`
- `DeepSpeed ZeRO-2/3`
- `vllm_mode="colocate"` 或使用独立节点
- Slurm 脚本启动多节点训练
- 数据并行和 tensor parallel
- 自动 push 到 Hugging Face Hub

参考：

- [Open-R1 README: Training models](https://github.com/huggingface/open-r1/blob/main/README.md)
- [Open-R1 `grpo.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py)

### 你应该对比学习什么

作业里的 `grpo_train_loop` 本质上就是工业训练系统的“最小可运行原型”。

可以这样映射：

- `rollout batch` -> 推理服务的一次采样批
- `reward_fn` -> 验证服务/评估器
- `old_log_probs cache` -> 训练中间产物缓存
- `eval every N steps` -> 周期性 benchmark
- `2 GPUs` -> 最小版训练/推理解耦

所以，理解作业循环之后，再看 Open-R1 的分布式栈，会更容易理解为什么系统设计会长成那个样子。

## 7. Reward 工程：Assignment 5 的简洁 reward 与 Open-R1 的组合式 reward

### Assignment 5 在做什么

Assignment 5 在数学任务上基本围绕两类信号：

- 格式正确
- 答案正确

并把更多精力放在：

- baseline
- normalization
- clipping
- entropy
- response length

这些影响训练稳定性和信用分配的地方。

### 工业界通常怎么做

工业界的 reward 往往不会只靠一个二值准确率，还会组合多个可操作信号：

- 准确率/正确性
- 格式奖励
- tag 奖励
- 长度惩罚
- 重复惩罚
- 代码执行分数
- overlong 惩罚
- reasoning 结构性奖励

原因很直接：

- 单一 reward 太稀疏，训练容易不稳定；
- 在真正的产品或 benchmark 环境中，输出格式和长度也很重要；
- 不同任务需要不同 verifier，例如数学解析器、单元测试、代码沙箱。

### Open-R1 是怎么做的

Open-R1 把 reward 做成注册表式函数集合，可以在配置里直接声明：

- 哪些 reward 生效；
- 每个 reward 的权重是多少；
- 某些 reward 用什么 provider，例如 `e2b`、`morph`、`piston`；
- 代码任务如何执行和评分。

这就是典型的“reward engineering 平台化”。

参考：

- [Open-R1 `rewards.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py)
- [Open-R1 GRPO demo config](https://github.com/huggingface/open-r1/blob/main/recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml)

### 你应该对比学习什么

如果说 Assignment 5 更像“用尽量少的 reward 讲清楚算法机制”，那 Open-R1 更像“面向真实任务的 reward 工程工具箱”。

对比学习时建议你问自己两个问题：

- 这个 reward 在作业里是算法变量，还是在工业里是产品/任务变量？
- 这个 reward 的代价来自数学定义，还是来自工程实现和外部服务延迟？

## 8. Prompt 与模板工程：作业里的 prompt ablation 对应工业里的 chat template 管理

### Assignment 5 在做什么

Assignment 5 明确要求比较：

- `r1_zero` prompt
- `question_only` prompt

它想让你看到：

- prompt 格式和模型预训练分布是否匹配，会显著影响 RL 效果。

### 工业界通常怎么做

工业界通常不会只把这件事叫做 “prompt ablation”，而会更系统地看待：

- chat template
- system prompt
- assistant prefill
- EOS token
- stop sequences
- 特定模型家族的模板兼容性

这些因素会直接影响：

- 训练格式 reward 是否生效；
- 思维链部分是否被正确保留；
- 推理后端能否正确停止；
- benchmark 时是否和训练格式一致。

### Open-R1 是怎么做的

Open-R1 在 README 和 recipe 中非常明确地强调：

- Qwen / DeepSeek 这类模型要特别小心 chat template 和 EOS token。
- 有些现成模板会把 `<think>` 区块隐藏掉，或者把 `<think>` 作为 prefill，进而干扰 format reward。
- 因此经常需要覆写 chat template。

参考：

- [Open-R1 README: chat template warning](https://github.com/huggingface/open-r1/blob/main/README.md)
- [Open-R1 GRPO demo config](https://github.com/huggingface/open-r1/blob/main/recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml)

### 你应该对比学习什么

在 Assignment 5 里，prompt 是研究变量。
在工业界里，template 是基础设施变量。

这是一个很值得建立的认知：

- “prompt engineering” 在 post-training 里并不只是写一句 system prompt，而是训练模板、生成模板、评测模板三者的一致性管理。

## 9. 消融实验：Assignment 5 的书面问题，工业界的 sweep 工作流

### Assignment 5 在做什么

Assignment 5 很系统地要求你做一系列消融：

- learning rate sweep
- baseline vs no baseline
- length normalization
- std normalization
- off-policy sweep
- clipping ablation
- prompt ablation

这本质上是教你如何像研究者一样看待训练系统。

### 工业界通常怎么做

工业界会把这件事做成 sweep 工作流：

- 使用 YAML 配置或实验配置系统统一记录每次运行参数。
- 用 W&B 或内部平台管理多次实验。
- 用 cheap eval 子集做早期判断，再决定是否上完整 benchmark。
- 统一汇总 reward、entropy、response length、grad norm、吞吐、显存使用等指标。
- 保留 best checkpoint 和失败 run 的日志，便于回溯。

### Open-R1 是怎么做的

Open-R1 的脚本是典型 recipe-driven 风格：

- CLI + YAML 配置
- W&B 记录
- callbacks
- benchmark 脚本
- 不同模型/任务各有 recipe

虽然它没有像作业文档一样逐条把所有 ablation 写成书面题，但它提供了适合做 sweep 的工程外壳。

参考：

- [Open-R1 `configs.py`](https://github.com/huggingface/open-r1/blob/main/src/open_r1/configs.py)
- [Open-R1 README](https://github.com/huggingface/open-r1/blob/main/README.md)

### 你应该对比学习什么

Assignment 5 教你的是“为什么要做这些 ablation”。
Open-R1 教你的是“怎样把 ablation 组织成可复现实验工程”。

## 10. 排行榜与最终优化：作业里的 2 卡预算，对应工业里的吞吐优化与系统并行

### Assignment 5 在做什么

作业最后要求你在固定预算下冲排行榜，并提示你思考：

- 为什么至少有一张 GPU 会空闲；
- 是否可以更好地并行 rollout 和训练；
- 是否可以用更低精度、`torch.compile`、系统优化等方法。

这一步已经开始从算法作业向系统作业过渡。

### 工业界通常怎么做

工业界通常会把这类问题上升为“训练吞吐优化”：

- rollout 和训练异步并行；
- 单独扩展推理节点；
- 使用 BF16/FP8、FlashAttention、Liger、CUDA graph 等优化；
- 按任务选择 data parallel / tensor parallel / pipeline parallel；
- 在 reward 成本很高的场景下，把 verifier 独立成服务；
- 优化日志频率和 checkpoint 频率，减少 IO 干扰；
- 使用模型与数据 registry，让大规模实验结果可追踪。

### Open-R1 是怎么做的

Open-R1 的工程答案基本都能在 README 中找到对应物：

- `DeepSpeed ZeRO-2/3`
- `vllm_mode="colocate"` 或独立 vLLM 节点
- Slurm 多节点
- 数据并行与 tensor parallel
- `flash_attention_2`
- `use_liger_kernel`
- `bf16`

这些都属于工业化训练栈的标准部件。

参考：

- [Open-R1 README: Training models](https://github.com/huggingface/open-r1/blob/main/README.md)
- [Open-R1 SFT recipe](https://github.com/huggingface/open-r1/blob/main/recipes/OpenR1-Distill-7B/sft/config_distill.yaml)

### 你应该对比学习什么

这一步最重要的学习不是“记住某个优化开关”，而是理解：

- 在 post-training 里，算法收益和系统吞吐是绑定的。

很多时候你能否做出更好的模型，取决于你是否能：

- 更快生成 rollout；
- 更便宜评估 reward；
- 更稳定恢复训练；
- 更系统地比较实验结果。

## 11. 一条推荐的对比学习路线

如果你的目标是同时理解 Assignment 5 的机制和工业工程流水线，推荐按下面的顺序学习：

### 第一阶段：先用 Assignment 5 吃透原理

- 手写并通过 SFT/GRPO 关键函数测试。
- 跑出最小可用的 zero-shot、SFT、GRPO 结果。
- 理解 `response_mask`、`old_log_probs`、`clip loss`、`entropy`、`group normalization`。

### 第二阶段：再用 Open-R1 建立工程映射

- 读 `sft.py`，把作业里的 SFT 原语映射到 `SFTTrainer`。
- 读 `grpo.py`，把作业里的 GRPO 原语映射到 `GRPOTrainer`。
- 读 `rewards.py`，把作业里的简单数学 reward 映射到 reward registry。
- 读 recipe YAML，理解 industrial run 是怎么被配置和调度的。

### 第三阶段：重点看 Assignment 5 中 Open-R1 没有直接暴露的部分

- Expert Iteration：理解它如何映射到 `generate + filter + retrain` 数据闭环。
- normalization / baseline / clipping 的消融：这是 Assignment 5 很强的教学价值。
- prompt ablation：映射到工业里的 chat template / EOS / prefill 管理。

### 第四阶段：尝试自己做一个“中间态实现”

一个很好的练习是：

- 保留 Assignment 5 的 reward 分析和实验视角；
- 但把 SFT 或 GRPO 改成用 `TRL + Accelerate + vLLM` 跑；
- 写一份自己的 recipe 和实验记录。

这会让你同时具备：

- 手写实现的算法理解；
- 框架实现的工业语感。

## 12. 最终结论

如果用一句话总结：

- Assignment 5 最适合学习“算法内部机制”。
- Open-R1 最适合学习“工业化训练流水线”。

两者最好的对比学习方式不是二选一，而是建立映射：

- `手写原语` -> `Trainer 抽象`
- `本地脚本` -> `配置驱动流水线`
- `单任务数学实验` -> `多任务 reasoning 平台`
- `显式 EI` -> `生成-过滤-再训练数据闭环`
- `2 卡 rollout/train loop` -> `多节点 vLLM + DeepSpeed 训练系统`

当你把这张映射表建立起来时，你对 reasoning post-training 的理解就会同时覆盖：

- 数学原理
- 训练实现
- 数据闭环
- 评测系统
- 分布式工程

这也是 Assignment 5 和 Open-R1 最适合一起读的原因。

## 参考资料

### Assignment 5

- 本地文档：`cs336_spring2025_assignment5_alignment_zh.md`

### Open-R1

- 总览与训练/评测说明：[README.md](https://github.com/huggingface/open-r1/blob/main/README.md)
- SFT 训练脚本：[src/open_r1/sft.py](https://github.com/huggingface/open-r1/blob/main/src/open_r1/sft.py)
- GRPO 训练脚本：[src/open_r1/grpo.py](https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py)
- 奖励函数注册表：[src/open_r1/rewards.py](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py)
- 配置定义：[src/open_r1/configs.py](https://github.com/huggingface/open-r1/blob/main/src/open_r1/configs.py)
- SFT recipe：[recipes/OpenR1-Distill-7B/sft/config_distill.yaml](https://github.com/huggingface/open-r1/blob/main/recipes/OpenR1-Distill-7B/sft/config_distill.yaml)
- GRPO demo recipe：[recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml](https://github.com/huggingface/open-r1/blob/main/recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml)
- 数据生成脚本：[src/open_r1/generate.py](https://github.com/huggingface/open-r1/blob/main/src/open_r1/generate.py)
- Pass-rate filtering：[scripts/pass_rate_filtering/README.md](https://github.com/huggingface/open-r1/blob/main/scripts/pass_rate_filtering/README.md)

> 说明：Open-R1 仓库会持续演化，这份笔记基于主分支公开内容整理，适合作为“Assignment 5 -> 工业工程实现”的对照学习地图。
