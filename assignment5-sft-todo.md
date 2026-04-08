**你的实验背景**
- 你的主线作业是 [cs336_spring2025_assignment5_alignment_zh.md](/home/nic/wyz/assignment5-alignment/cs336_spring2025_assignment5_alignment_zh.md) 里的 `Qwen2.5-Math-1.5B` 在 `MATH` 上的 `zero-shot baseline -> SFT -> Expert Iteration -> GRPO`。
- 你已经基本完成了除训练脚本外的核心实现，重点 helper functions 在 `tokenization / modeling / losses / rewards` 这条线上已经打通。
- 你本地机器是 `RTX TITAN 24GB`，Turing 架构，不能稳定使用 `bf16` 和 `flash-attention-2`，也没有安装 `flash-attn==2.7.4.post1`。
- 这意味着本地更适合做 `smoke test` 和小样本调试，不适合直接复刻 handout 里的正式 SFT 配置。
- 4.3 的正式 SFT 目标是：周期性在 `MATH validation` 上评估，理想配置是双卡，一张放 policy，一张放 vLLM 做验证生成。
- 4.3 的最终交付物不是“脚本能跑”而已，而是两类结果：`{128,256,512,1024,full}` 数据规模曲线，以及 `filtered-correct SFT` 的数据规模和验证曲线。
- 你的下一阶段最重要的任务不是继续补底层函数，而是把 [cs336_alignment/sft.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/sft.py) 变成一个可执行、可切换本地/云上配置的实验入口。

**SFT 实验方案**
1. 先完成 `run_sft_experiment` 的“本地 smoke + 云上正式”双配置能力。至少要支持切换 `torch_dtype`、`attn_implementation`、`train_device`、`eval_device`、`max_train_examples`、`eval_num_examples`、`train_batch_size`、`gradient_accumulation_steps`。
2. 本地 smoke 模式用 `fp16 + eager/sdpa`，不要默认 `bf16 + flash_attention_2`。你的本地卡不适合后者。
3. 本地 smoke 时先不要强求双卡 vLLM 评测。可以先把 `eval_device=None`，或者只做极小验证子集。
4. 本地 smoke 的目标不是冲指标，而是确认训练链路完整：读数据、前向、loss、backward、保存 checkpoint、最小评测都正常。
5. 云上正式模式再切回 handout 推荐配置：`bf16 + flash_attention_2`，训练卡和评测卡分开，定期在至少 `1024` 个验证样本上评估。
6. 正式 SFT 先跑数据规模扫描：`128 / 256 / 512 / 1024 / full`，每个配置记录 train/eval 曲线。
7. 之后再做 filtered-correct SFT：用 `r1_zero_reward_fn(response, ground_truth)["answer_reward"] == 1.0` 过滤 `sft.jsonl`，记录过滤后数据集大小，再跑 full filtered SFT。
8. 每个实验都保存 `history.json`、`best checkpoint`、`final checkpoint`、关键超参数，避免后面画图和写报告时回头补数据。
9. 先完成 SFT，再做 EI；先完成 EI，再做 GRPO。不要跳步骤。

**建议的本地 smoke 配置**
- `torch_dtype=torch.float16`
- `attn_implementation="sdpa"` 或 `"eager"`
- `max_train_examples=32` 或 `64`
- `eval_num_examples=16` 或 `32`
- `num_epochs=1`
- `train_batch_size=2` 或 `4`
- `gradient_accumulation_steps=4` 或 `8`
- 如果要生成验证输出，先把 `max_tokens` 压到 `256` 或 `512`
- 只要能看到 loss 正常、checkpoint 正常、最小验证能返回结果，这一步就算成功

**建议的云上正式配置起点**
- `torch_dtype=torch.bfloat16`
- `attn_implementation="flash_attention_2"`
- `train_device="cuda:0"`
- `eval_device="cuda:1"`
- `eval_every_steps=50` 或 `100`
- `eval_num_examples=1024`
- `max_grad_norm=1.0`
- 学习率先扫 `5e-6 / 1e-5 / 2e-5`
- 有效 batch size 先从 `32` 或 `64` 起步，不要一开始就贪大
- 每个配置结束后再跑一次 full validation，用于最后报告

**To Do List**
1. 检查 [cs336_alignment/modeling.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/modeling.py) 是否支持本地 `fp16 + sdpa/eager` 与云上 `bf16 + flash_attention_2` 切换。
2. 完成 [cs336_alignment/sft.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/sft.py) 的训练入口，让它支持本地 smoke 和云上正式两套参数。
3. 在 `sft.py` 里接入周期性验证评测，并保存 train/eval 历史。
4. 加入 `best checkpoint` 保存逻辑，不要只存最终模型。
5. 先在本地跑一次 `32` 或 `64` 条样本的 smoke test。
6. 修掉 smoke test 里暴露出来的路径、dtype、显存、保存格式问题。
7. 准备云上正式实验配置，单独写一份参数表，不和本地 smoke 混用。
8. 先跑 `128 / 256 / 512 / 1024 / full` 的 SFT 数据规模实验。
9. 汇总验证准确率曲线，选一个最稳的学习率和 batch 方案。
10. 构造 filtered-correct 数据集，记录过滤后样本数。
11. 跑 filtered-correct SFT，并和未过滤 full SFT 对比。
12. 整理结果目录、画图、更新 [writeup.md](/home/nic/wyz/assignment5-alignment/writeup.md)。

**你现在最该做的一步**
- 直接开始改 [cs336_alignment/sft.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/sft.py)，目标不是一次做完所有实验，而是先让它支持“本地 smoke 模式”稳定跑通。

如果你愿意，我下一步可以直接帮你把这份 plan 进一步细化成“今天就能执行的 2 小时任务清单”。

**今天可执行的 2 小时任务清单**

1. 检查 [cs336_alignment/modeling.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/modeling.py)  
   目标：把 `load_policy_and_tokenizer()` 改成支持本地与云上两套配置。
   你要确认：
   - 本地可传 `torch_dtype=torch.float16`
   - 本地可传 `attn_implementation="sdpa"` 或 `"eager"`
   - 云上还能切回 `bfloat16 + flash_attention_2`

2. 检查 [cs336_alignment/sft.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/sft.py)  
   目标：让它能以“本地 smoke 模式”跑起来。
   今天先不追求全功能，只确保支持：
   - `max_train_examples`
   - `eval_num_examples`
   - `train_batch_size`
   - `gradient_accumulation_steps`
   - `eval_device=None`

3. 设定一组本地 smoke 参数  
   建议今天先固定成：
   - `max_train_examples=32`
   - `eval_num_examples=16`
   - `num_epochs=1`
   - `train_batch_size=2`
   - `gradient_accumulation_steps=2` 或 `4`
   - `torch_dtype=float16`
   - `attn_implementation=sdpa`
   - 不开 vLLM 双卡验证，或只保留极小验证

4. 跑一次最小训练  
   目标不是看指标，而是确认：
   - 数据加载没问题
   - 前向和 backward 正常
   - loss 能输出
   - 模型能保存
   - 历史日志能写盘

5. 如果最小训练报错，按优先级修  
   优先排查：
   - dtype / device 不匹配
   - attention implementation 不兼容
   - 显存不足
   - tokenizer / pad token 问题
   - 保存目录 / 路径问题

6. 跑通后记录一份“本地 smoke 成功配置”  
   把今天能跑通的参数单独记下来，后面不要反复试。

---

**今天的完成标准**
- [cs336_alignment/sft.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/sft.py) 能在本地用小样本跑完一轮
- 能看到至少一个保存出来的 `history` 或 checkpoint
- 你能明确区分“本地 smoke 参数”和“云上正式参数”

---

**下一步接续计划**
- 今天跑通本地 smoke
- 下一步补上更完整的 eval / best checkpoint / 数据规模扫描
- 再准备云上正式 SFT

如果你愿意，我下一步可以直接开始帮你把 [cs336_alignment/modeling.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/modeling.py) 和 [cs336_alignment/sft.py](/home/nic/wyz/assignment5-alignment/cs336_alignment/sft.py) 改成适合你本地 smoke 的版本。