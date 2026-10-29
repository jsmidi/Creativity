"""Generation engine for locally downloaded Transformers causal language models."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class GenerationEngine:
    """Small generation-only wrapper; it has no activation intervention hooks."""

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", dtype="auto", local_files_only=True
        ).eval()

    @torch.inference_mode()
    def generate(self, prompt, seed=42, max_new_tokens=800, temperature=0.7, top_p=1.0):
        if self.tokenizer.chat_template:
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        options = {"max_new_tokens": max_new_tokens, "use_cache": True}
        if self.tokenizer.eos_token_id is not None:
            options["pad_token_id"] = self.tokenizer.eos_token_id
        if temperature > 0:
            options.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            options["do_sample"] = False
        output = self.model.generate(**inputs, **options)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return {
            "Response": self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            "Generated_Tokens": new_tokens.numel(),
            "Prompt_Tokens": inputs["input_ids"].shape[1],
        }

