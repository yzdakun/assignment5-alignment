from vllm import LLM, SamplingParams

prompts = [
    "What is 1 + 1?",
    "What is the capital of France?",
]

sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=128,
)

llm = LLM(model="../models/Qwen2.5-Math-1.5B", dtype = "float16")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print("prompt:", output.prompt)
    print("text:", output.outputs[0].text)
