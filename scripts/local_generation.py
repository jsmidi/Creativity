"""Behavioral model loading; GPT-OSS does not use the Llama intervention adapter."""
import re
from sampling import sampling_kwargs
import torch


def final_answer(raw):
    """Extract only an assistant final channel; never score analysis as an answer."""
    match = re.search(r"(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>(.*)", raw, re.S)
    if not match:
        return ""
    return re.split(r"<\|(?:return|fim_suffix|end|start|ghissue|handoff)\|>", match[1], maxsplit=1)[0].strip()


class GptOssEngine:
    def __init__(self, model, tokenizer, reasoning_effort):
        """Store an evaluation-mode GPT-OSS model, tokenizer and requested reasoning effort."""
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.reasoning_effort = reasoning_effort

    @torch.inference_mode()
    def generate(self, prompt, seed=42, temperature=0.7, max_tokens=800, top_p=1.0, top_k=0):
        """Seed generation and return the final answer separately from raw reasoning and token counts."""
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], reasoning_effort=self.reasoning_effort,
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.model.get_input_embeddings().weight.device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        stop = self.tokenizer.eos_token_id
        kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature > 0,
                      eos_token_id=stop, pad_token_id=self.tokenizer.eos_token_id,
                      use_cache=True)
        kwargs.update(sampling_kwargs(temperature, top_p, top_k))
        tokens = self.model.generate(**inputs, **kwargs)
        completion = tokens[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(completion, skip_special_tokens=False)
        return dict(Response=final_answer(raw), Raw_Completion=raw,
                    Reasoning_Effort=self.reasoning_effort, Quantization="dequantized_bfloat16",
                    Generated_Tokens=completion.numel(), Prompt_Tokens=inputs["input_ids"].shape[1],
                    Finish_Reason="stop" if completion.numel() and completion[-1].item() == stop else "length",
                    Intervention_Calls=0, Forward_Calls=0)


def gpt_oss_memory_budget():
    """Reserve 8 GiB per GPU for activations, KV cache, and loading overhead."""
    budgets = {}
    for device in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(device)
        usable = free // 1024**3 - 8
        if usable > 0:
            budgets[device] = f"{usable}GiB"
    if sum(int(value[:-3]) for value in budgets.values()) < 60:
        raise RuntimeError(
            "Insufficient GPU memory for BF16 GPT-OSS. Request two 40GB A100s: "
            "sbatch --gpus=2 scripts/snellius_behavioral.sh gpt-oss "
            "(or use one 80GB GPU). At least 60 GiB of free GPU memory after "
            "reserving 8 GiB per device is required.")
    return budgets


def load_behavioral_engine(model_id, reasoning_effort="low"):
    """Load the local behavioral backend; GPT-OSS uses BF16 memory checks and final-channel parsing."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    config = AutoConfig.from_pretrained(model_id)
    if config.model_type != "gpt_oss":
        from interventions import load_engine
        return load_engine(model_id)
    if not torch.cuda.is_available():
        raise RuntimeError("Run GPT-OSS inside a GPU allocation.")
    memory = gpt_oss_memory_budget()
    print(f"GPT-OSS BF16 GPU weight budgets: {memory}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", max_memory={**memory, "cpu": 0}, dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True), attn_implementation="sdpa")
    if any(str(device) in ("cpu", "disk") for device in model.hf_device_map.values()):
        raise RuntimeError("GPT-OSS spilled to CPU/disk; request more GPU memory.")
    print(f"GPT-OSS device map: {model.hf_device_map}", flush=True)
    return GptOssEngine(model, tokenizer, reasoning_effort)
