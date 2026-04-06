# CS336 Assignment 5 混合学习与实验方案

更新日期：2026-04-01

## 1. 目标与约束

你的目标不是“最低成本硬跑完所有东西”，而是：

1. 最大化理解这份作业里从 `zero-shot baseline -> SFT -> Expert Iteration -> GRPO` 的完整链路。
2. 尽量把“需要人脑和代码思考”的部分放在本地完成，把“需要长时间稳定 GPU”的部分集中到云上。
3. 只在真正会产生知识增量的节点租用 AutoDL，避免把云上时间浪费在环境配置、报错排查、重复试错上。

你的现实约束是：

- 本地：`RTX TITAN 24GB`
- 云上目标卡：`RTX PRO 6000 96GB`
- 本仓库主线实验围绕 `Qwen2.5-Math-1.5B`，不是 7B/14B/70B 主模型
- 可选安全评测脚本 `scripts/evaluate_safety.py` 使用 `Llama-70B`，成本明显更高，不应纳入默认计划

## 2. 先给结论

推荐采用“本地开发 + 云上短时集中执行”的两阶段方案：

- 本地完成：读题、代码实现、单元测试、reward/评测链路理解、小样本 smoke test、baseline 可行性验证
- 云上完成：正式 SFT、Expert Iteration、GRPO，以及必要时的完整 baseline 重跑
- 默认只租 `1 x RTX PRO 6000 96GB`
- 只有在你真的要做可选安全部分时，才考虑 `2 x RTX PRO 6000 96GB`

这样做的核心原因是：

- 这个项目最值钱的学习内容，其实是 helper functions、loss、reward、rollout、evaluation 的实现与调试，而不是“把卡开着等训练结束”
- 你本地 24GB 足够支持绝大多数开发与调试工作
- 云上真正值得花钱的，是“已经准备好后的一次性正式实验窗口”

## 3. 任务拆分：哪些放本地，哪些放云上

| 模块 | 推荐地点 | 原因 | 是否必须上云 |
| --- | --- | --- | --- |
| 阅读作业 handout、中文翻译、理解实验流程 | 本地 | 不需要 GPU | 否 |
| 实现 `tokenization.py`、`modeling.py`、`losses.py`、`rewards.py` 的核心函数 | 本地 | 主要是代码与单测驱动 | 否 |
| 运行 `tests/test_sft.py`、`tests/test_grpo.py` | 本地 | 仓库自带 tiny fixture，成本低 | 否 |
| baseline 小样本验证 | 本地 | 1.5B 模型在 24GB 上大概率可做小规模推理 | 否 |
| baseline 全量 5000 条验证集 | 优先本地，失败再上云 | 这是纯推理任务，适合夜跑；如果本地 vLLM 不稳再放云上 | 不一定 |
| SFT 小样本 smoke run | 本地优先，必要时云上 | 学习价值高，但正式跑完整实验更适合云上 | 不一定 |
| 正式 SFT 实验 | 云上 | 需要更稳定显存和更长时段运行 | 是 |
| Expert Iteration | 云上 | 需要反复 rollout + filter + train，时长更长 | 是 |
| GRPO | 云上 | rollout + scoring + policy update 对显存和时长要求更高 | 是 |
| 可选 safety / RLHF / 70B judge | 默认不做 | 成本显著上升，主线学习收益不成比例 | 否 |

## 4. 省钱原则

### 4.1 总原则

把云上时间只花在这三类事情上：

1. 本地无法稳定完成的正式实验
2. 必须依赖大显存的长时运行
3. 已经确认代码正确后的结果产出

不要把云上时间花在这三类事情上：

1. 环境安装试错
2. 单元测试修 bug
3. 小样本逻辑验证

### 4.2 AutoDL 计费与省钱要点

根据 AutoDL 官方文档，我在 2026-04-01 查到的关键信息是：

- 按量计费是“开机开始计费，关机结束计费”，所以不使用时必须关机
- 包年包月支持按日/按周/按月，价格通常比按量更便宜
- 官方文档明确提到：更经济的用法是“先包年包月，再在不需要时转按量退款处理”
- 优惠券仅适用于包年包月，不适用于直接按量
- 实例关机后数据和环境会保留，但连续关机 15 天实例会被释放
- 同一地区网盘默认有 20GB 免费空间，可用于多实例之间共享文件
- 镜像可保存系统盘环境，避免重复配环境
- 官方价格文档没有固定写死每张卡的单价，而是明确说明“请以网站显示的现价为准”

这意味着你最推荐的租用策略是：

- 预计一次连续实验窗口超过半天时：优先考虑“包日 + 优惠券 + 提前结束后转按量退款”
- 只是 1 到 3 小时的短时冒烟：按量通常更灵活
- 无论哪种计费方式，只要当天不用就立刻关机

## 5. 推荐的总体执行方案

我推荐你采用下面这个“学习优先、同时尽量省钱”的方案。

### 阶段 A：本地打基础，不上云

目标：

- 完成仓库结构理解
- 把必做 helper functions 补起来
- 跑通单元测试
- 对 baseline / reward / logging 流程建立清晰认知

重点文件：

- `cs336_alignment/tokenization.py`
- `cs336_alignment/modeling.py`
- `cs336_alignment/losses.py`
- `cs336_alignment/rewards.py`
- `cs336_alignment/evaluate.py`
- `cs336_alignment/sft.py`
- `cs336_alignment/expert_iteration.py`
- `cs336_alignment/grpo.py`
- `tests/adapters.py`

本地建议顺序：

1. 先读 `README.md`、`assignment5.txt`、`cs336_spring2025_assignment5_alignment_zh.md`
2. 再实现并通过：
   - `test_tokenize_prompt_and_output`
   - `test_compute_entropy`
   - `test_get_response_log_probs`
   - `test_masked_normalize`
   - `test_sft_microbatch_train_step`
   - `test_compute_group_normalized_rewards`
   - `test_compute_naive_policy_gradient_loss`
   - `test_compute_grpo_clip_loss`
   - `test_compute_policy_gradient_loss`
   - `test_grpo_microbatch_train_step`
3. 最后再接高层实验入口

本地完成标志：

- 必做测试稳定通过
- 你能口头解释清楚：response mask 是什么、reward 如何计算、SFT 和 GRPO loss 差异在哪里
- 你至少完成一次 50 到 200 条数据规模的 smoke test

### 阶段 B：本地小样本 smoke test

目标：

- 确认完整链路不是“单测通过但脚本跑不起来”
- 提前暴露 I/O、路径、prompt、序列长度、保存结果等问题

建议你本地只做“小样本 + 短输出 + 少步骤”的验证：

- baseline：先跑一个小子集
- SFT：只跑少量 step，确认 loss 下降、模型能保存
- Expert Iteration：只跑 1 个 iteration
- GRPO：只跑 1 个 rollout batch + 极少 train step

这一阶段的意义不是看最终指标，而是验证：

- 数据读取正确
- 模型 / tokenizer 加载正确
- vLLM 与 HF policy 之间的数据流正确
- reward 输出正确
- 中间结果能正确落盘

如果你的本地 `RTX TITAN 24GB` 能稳定完成完整 baseline，那么正式 baseline 也建议留在本地夜跑，进一步省云上费用。

### 阶段 C：云上环境固化

目标：

- 只做一次环境安装
- 之后所有正式实验都复用同一镜像

推荐做法：

1. 在 AutoDL 先租一次短时实例
2. 拉代码、装依赖、验证 GPU / vLLM / FlashAttention / `uv` 是否正常
3. 跑最小 smoke test
4. 立刻保存镜像
5. 以后正式实验都从这个镜像启动

这样做的收益非常大：

- 后续每次开机不用重新装环境
- 降低“今天租了卡，但前两小时都在装包”的浪费
- 方便你按实验阶段拆成多个短租窗口

### 阶段 D：云上正式实验

默认只用 `1 x RTX PRO 6000 96GB`，按下面顺序跑：

1. baseline（仅当本地完整 baseline 不稳时）
2. SFT
3. Expert Iteration
4. GRPO

顺序不要反：

- baseline 是参照系
- SFT 是最好的 warm-up 学习实验
- EI 比 GRPO 更容易帮助你理解“过滤正确轨迹再训练”的思想
- GRPO 最复杂，也最值得在你前面都跑通以后再做

## 6. 详细学习时间表

下面给你一份“推荐版 10 天计划”。如果你推进更快，也可以压缩成 5 到 7 天。

### 第 1 天：读题与画图

任务：

- 通读 `README.md`
- 通读 `assignment5.txt`
- 对照中文翻译看整体流程
- 用一页纸画出以下链路：
  - prompt -> tokenize -> model logits -> log_probs -> loss
  - prompt -> vLLM rollout -> reward -> filter / advantage -> update

产出：

- 一份你自己的流程图
- 一份你不理解的问题清单

### 第 2 天：SFT 基础函数

任务：

- 完成 tokenization、entropy、response log-probs
- 跑通相关测试

你需要真正掌握：

- 为什么 `labels` 比原序列左移 1
- 为什么 response mask 对训练至关重要
- 为什么 log-softmax 比直接 softmax 再 log 更稳

产出：

- helper functions 通过测试

### 第 3 天：SFT 训练步

任务：

- 完成 `masked_normalize`
- 完成 `sft_microbatch_train_step`
- 跑通 `test_sft.py`

你需要真正掌握：

- gradient accumulation 在这里为什么必要
- microbatch loss 与 effective batch size 的关系

产出：

- SFT 训练步稳定通过测试

### 第 4 天：GRPO 基础函数

任务：

- 完成 `compute_group_normalized_rewards`
- 完成 policy gradient / GRPO clip loss
- 跑通 `test_grpo.py`

你需要真正掌握：

- raw reward 与 advantage 的区别
- group normalization 在这里扮演什么角色
- clipping 为什么能稳定训练

产出：

- GRPO 相关测试通过

### 第 5 天：高层脚本与本地 smoke run

任务：

- 把 `evaluate.py`、`sft.py`、`expert_iteration.py`、`grpo.py` 串起来
- 本地做小样本 smoke run

产出：

- 每个实验至少有一次最小运行记录
- 明确哪些实验本地能跑、哪些必须上云

### 第 6 天：云上环境固化日

任务：

- 短租 1 张 `RTX PRO 6000 96GB`
- 安装环境并保存镜像
- 复测最小 smoke run

产出：

- 可复用镜像
- 一套稳定命令清单

### 第 7 天：正式 baseline + SFT

任务：

- 如果 baseline 还没在本地完整跑完，补跑
- 正式跑 SFT
- 记录训练曲线、验证结果、样例输出

产出：

- baseline 结果
- SFT 模型与评估结果

### 第 8 天：Expert Iteration

任务：

- 先跑 1 轮小规模
- 确认过滤出来的“正确轨迹”质量
- 再决定是否扩大规模

产出：

- 至少一组 EI 结果
- 对“生成 -> 验证 -> 过滤 -> 再训练”机制的直观理解

### 第 9 天：GRPO

任务：

- 先保守参数跑通
- 再增加 rollout / train step
- 不要一开始就追求最大规模

产出：

- 至少一组 GRPO 结果
- 对 advantage / clipping / entropy 的观察

### 第 10 天：结果整理与 writeup

任务：

- 整理表格、图、关键样例
- 对比 baseline / SFT / EI / GRPO
- 写出你真正学到的东西

重点不是“哪个数字最高”，而是：

- 为什么某个方法有效
- 为什么某个方法在小模型上效果有限
- 哪些失败样例暴露了格式问题、答案问题、还是 reward 问题

## 7. 你应该如何安排云上租用窗口

### 推荐方案：两到三个短窗口

#### 窗口 1：环境与冒烟

目标：

- 配好环境
- 保存镜像
- 不追求正式结果

建议时长：

- 1 到 3 小时

计费建议：

- 预估只需要 1 到 3 小时就按量
- 如果你还要顺手跑一次正式 baseline，可考虑包日

#### 窗口 2：SFT + baseline

目标：

- 出第一批正式结果

建议时长：

- 半天到一天

计费建议：

- 更偏向包日
- 有优惠券优先用包日

#### 窗口 3：EI + GRPO

目标：

- 完成主线 RL 实验

建议时长：

- 一天左右，取决于你设定的 rollout 数、train step 数和评测频率

计费建议：

- 更偏向包日
- 提前做完后按官方规则转按量退款

### 为什么不推荐“一次租很多天”

因为真正烧钱的不是训练本身，而是下面这些碎片损耗：

- 中途发现脚本 bug
- 结果写盘有问题
- 参数配错需要重跑
- 当天状态不好，实验结束后没及时关机

对于个人学习者，短租窗口比长租窗口通常更可控。

## 8. 云上执行剧本

每次开机前都照这个顺序来，能明显省钱。

### 8.1 开机前 checklist

- 本地代码已经提交到 git 或至少本地打包备份
- 你今天要跑的实验清单已经写好
- 你今天要记录的指标已经定义好
- 你知道“跑到什么条件就停止”
- 你已经决定今天是按量还是包日

### 8.2 开机后前 10 分钟

- 先确认 GPU、CUDA、`uv`、PyTorch、vLLM 是否正常
- 先跑一个最小 smoke test
- 如果 smoke test 不通过，立刻停下来修，不要直接开正式实验

### 8.3 正式运行时

- 每次只跑一类实验，不要多个大任务同时开
- 结果目录按日期和实验名分开
- 先小规模，再扩大
- 评测频率不要过高，避免把大量时间耗在重复验证上

### 8.4 关机前

- 保存结果
- 备份关键日志
- 保存模型路径和配置
- 需要的话保存镜像
- 确认实例已关机
- 如果扩容了数据盘，评估是否需要缩容

## 9. 参数与实验策略建议

### 9.1 baseline

建议：

- 先用一个固定小子集做 smoke test
- 完整 5000 条验证集只在脚本稳定后跑一次
- 不要因为格式问题重跑很多次，先分析 reward 失败样例

学习重点：

- 理解 prompt 对输出格式的影响
- 理解 parser / reward 与模型行为的关系

### 9.2 SFT

建议：

- 先追求“训练流程正确”，再追求更大 batch
- 在 96GB 卡上也优先从保守 microbatch 开始
- 只在你确认 loss、保存、评测都正常后再扩大训练规模

学习重点：

- response-only loss
- gradient accumulation
- sequence length 与显存关系

### 9.3 Expert Iteration

建议：

- 先做较小 `G` 和较少 epoch
- 观察过滤后样本量，不要盲目扩大 rollout
- 如果正确轨迹极少，先排查 reward 和 prompt，再加大计算

学习重点：

- verified reward 如何把“生成”转化成“可训练样本”
- 为什么 EI 更像“bootstrapping”

### 9.4 GRPO

建议：

- 从最保守的 rollout batch 与 train step 起步
- 先观察 reward 分布、advantage 分布、entropy 变化
- 不要一开始就追求最大 clip / 最大 steps

学习重点：

- policy gradient 的不稳定性
- group-relative normalization 的作用
- entropy 与探索-收敛的关系

## 10. 为了省钱，你应该主动放弃什么

下面这些内容默认不建议投入云上预算：

- 可选的 safety / RLHF 分支完整复现
- `Llama-70B` 安全 judge 的完整跑通
- 多卡扩张实验
- 很多组超参数网格搜索
- 为了“更好看”的数字做大量重复实验

更好的做法是：

- 主线实验每类拿到 1 到 2 组可信结果
- 把精力放在理解机制和分析失败样例
- writeup 中写清楚你的判断与 trade-off

## 11. 预算控制模板

由于 AutoDL 官方价格文档明确说明“请以网站显示的现价为准”，这里不写死单价，只给你预算公式。

设：

- `P96` = AutoDL 上 `1 x RTX PRO 6000 96GB` 当前显示的小时价或日价
- `D` = 数据盘/文件存储/镜像超额费用
- `N1` = 窗口 1 用时
- `N2` = 窗口 2 用时
- `N3` = 窗口 3 用时

则：

- 按量估算：`总费用 ≈ P96 * (N1 + N2 + N3) + D`
- 包日估算：`总费用 ≈ 包日单价 * 天数 + D - 可退款部分 - 优惠券减免`

### 我的推荐预算策略

#### 方案 A：最均衡，推荐

- 本地完成全部开发与单测
- baseline 优先本地跑
- 云上只跑 SFT / EI / GRPO
- 云上总共分 2 到 3 个短窗口

适合你现在的目标：学习最大化、费用可控、风险低。

#### 方案 B：极限省钱

- 本地完成全部开发
- 本地完成 baseline
- 云上只跑 SFT 和 GRPO
- Expert Iteration 只做小规模或只写设计不正式跑全量

适合预算非常紧张时。

#### 方案 C：结果优先

- baseline、SFT、EI、GRPO 全部云上跑
- 同一张 96GB 卡连续开一天或两天

适合你已经非常确定代码正确，只想快速拿结果时。对学习和费用都不是最优。

## 12. 你现在最应该做的三件事

1. 先在本地把 `tests/test_sft.py` 和 `tests/test_grpo.py` 全部打通。
2. 本地做 1 次完整小样本 smoke run，确认高层脚本是通的。
3. 只有在这两件事都完成后，再去租第一张 AutoDL 的 `RTX PRO 6000 96GB`。

如果这三步顺序反了，你大概率会把钱花在调 bug 上，而不是花在学习上。

## 13. 参考资料

### 课程与仓库

- 仓库说明：`README.md`
- 作业原文：`assignment5.txt`
- 中文翻译：`cs336_spring2025_assignment5_alignment_zh.md`
- 可选安全评测脚本：`scripts/evaluate_safety.py`

### 官方资料

- AutoDL 计费说明：https://www.autodl.com/docs/price/
- AutoDL 快速开始：https://www.autodl.com/docs/quick_start/
- AutoDL GPU 选型：https://www.autodl.com/docs/gpu/
- AutoDL 优惠券说明：https://www.autodl.com/docs/cp/
- AutoDL 网盘说明：https://www.autodl.com/docs/nas/
- AutoDL 镜像说明：https://www.autodl.com/docs/image/
- AutoDL 保存镜像：https://www.autodl.com/docs/save_image/
- NVIDIA RTX PRO 6000 Blackwell 96GB 官方规格：https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/
- NVIDIA RTX 6000 Ada 48GB 官方规格：https://www.nvidia.com/en-us/products/workstations/rtx-6000/

## 14. 一句话总结

最优解不是“尽量少租卡”，而是“把云上时间只留给已经准备好的正式实验”。先在本地把代码、测试、smoke run 全做扎实，再用 `1 x RTX PRO 6000 96GB` 分两到三个短窗口集中产出结果，这样最能兼顾学习深度、完成度和租用成本。
