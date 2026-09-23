from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("agcbench-2026/AGC-Judge")
model = AutoPeftModelForCausalLM.from_pretrained(
    "agcbench-2026/AGC-Judge",
    torch_dtype="auto",
    device_map="auto",
)

prompt = (
    "Benchmark rubric:\n{rubric}\n\n"
    "Prompt:\n{instruction}\n\nResponse:\n{response}\n\n"
    "Output a single integer score on the scale specified in the rubric. "
    "No explanation, no formatting, just the number.\n\nScore:"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=4)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
