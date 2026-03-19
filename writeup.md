# 3
## problem: math_baseline
(a) A script to evaluate baseline zero-shot MATH performance
{
  "num_examples": 5000,
  "avg_reward": 0.0226,
  "avg_format_reward": 0.1622,
  "avg_answer_reward": 0.0226,
  "count_format1_answer1": 113,
  "count_format1_answer0": 698,
  "count_format0_answer0": 4189
}


(b)  Run your evaluation script on Qwen 2.5 Math 1.5B. How many model generations fall into each
of the following categories: (1) correct with both format and answer reward 1, (2) format reward
1 and answer reward 0, (3) format reward 0 and answer reward 0? Observing at least 10 cases
where format reward is 0, do you think the issue is with the base model’s output, or the parser?
Why? What about in (at least 10) cases where format reward is 1 but answer reward is 0?
Deliverable: Commentary on the model and reward function performance, including examples
of each category.
(1) 在观察 format_reward = 0 的样本后，我认为主要问题出在模型输出，而不是解析器。很多失败样本并没有按照要求生成完整、规范的 <think> ... </think> <answer> ... </answer> 结构，而是出现了标签缺失、标签错配、重复闭合、输出跑题、多个问题拼接、甚至混入无关代码和对话片段等现象。这说明 base model 在 R1-Zero prompt 下的格式服从性较差。虽然解析器在少数边界情况下可能较严格，但从这些样本来看，主导失败因素仍然是模型输出本身不规范。
(2)
对于 format_reward = 1 但 answer_reward = 0 的样本，模型通常已经能够生成符合要求的 <think> ... </think> <answer> ... </answer> 结构，因此这类错误主要不是格式问题，而是数学推理或题意理解问题。常见现象包括：中间推导错误、使用了错误公式、计数或几何计算出错、对题目条件理解有偏差，或者给出看似合理但实际上与标准答案不一致的最终答案。这说明即使模型能够遵守输出格式，其 zero-shot 条件下的数学求解能力仍然较弱。
(3)
对于 reward = 1 的样本，模型不仅生成了可解析的 <answer> ... </answer> 格式，而且其最终答案与标准答案在数学上等价，因此同时获得了格式奖励和答案奖励。值得注意的是，这并不要求 reasoning 一定完整或高质量；只要最终答案正确且格式可解析，grader 就会将其记为成功样本。

(c) How well does the Qwen 2.5 Math 1.5B zero-shot baseline perform on MATH?
Deliverable: 1-2 sentences with evaluation metrics.
Qwen2.5-Math-1.5B 在 MATH 验证集上的 zero-shot baseline 表现较差，平均 reward 和 answer_reward 都为 0.0226，即最终答案正确率约为 2.26%。同时，平均 format_reward 仅为 0.1622，说明模型在大多数情况下甚至不能稳定遵守 R1-Zero prompt 所要求的输出格式。