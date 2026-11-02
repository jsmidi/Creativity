"""Generation engine for locally downloaded Transformers causal language models."""
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def checkpoint_fingerprint(model_path):
    """SHA-256 fingerprint of local weights, model config, and tokenizer files."""
    root = Path(model_path)
    if not root.is_dir():
        return ""

    files = set()
    metadata_names = {
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "pytorch_model.bin.index.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
        "tokenizer.model", "spiece.model", "chat_template.jinja",
    }
    for name in metadata_names:
        candidate = root / name
        if candidate.is_file():
            files.add(candidate)

    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        index_path = root / "pytorch_model.bin.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        files.update(root / name for name in set(index.get("weight_map", {}).values()))
    else:
        files.update(root.glob("*.safetensors"))
        files.update(root.glob("pytorch_model*.bin"))

    files = sorted(path for path in files if path.is_file())
    if not files or not any(path.suffix in {".safetensors", ".bin"} for path in files):
        return ""

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return "sha256:" + digest.hexdigest()

class GenerationEngine:
    """Generation-only wrapper for locally downloaded Transformers models."""

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", dtype="auto", local_files_only=True
        ).eval()
        self.model_revision = (
            getattr(self.model.config, "_commit_hash", None)
            or getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
            or ""
        )
        print("Fingerprinting local model and tokenizer files...", flush=True)
        self.model_fingerprint = checkpoint_fingerprint(model_path)

    def _format_prompt(self, prompt):
        if self.tokenizer.chat_template:
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs.to(self.model.get_input_embeddings().weight.device)

    @torch.inference_mode()
    def generate(self, prompt, seed=42, max_new_tokens=800, temperature=0.7, top_p=1.0):
        inputs = self._format_prompt(prompt)
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
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple, set))
                      else ([] if eos_ids is None else [eos_ids]))
        if new_tokens.numel() and int(new_tokens[-1]) in eos_ids:
            finish_reason = "eos"
        elif new_tokens.numel() >= max_new_tokens:
            finish_reason = "max_new_tokens"
        else:
            finish_reason = "other"
        return {
            "Response": self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            "Generated_Tokens": new_tokens.numel(),
            "Prompt_Tokens": inputs["input_ids"].shape[1],
            "Prompt_Token_IDs": inputs["input_ids"][0].detach().cpu().tolist(),
            "Response_Token_IDs": new_tokens.detach().cpu().tolist(),
            "Finish_Reason": finish_reason,
        }

