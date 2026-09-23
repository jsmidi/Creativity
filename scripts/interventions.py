"""Local causal interventions for Llama-style decoders, including true head slices.

Head coordinates are BEFORE o_proj, not slices of its mixed output. Residual
coordinates are AFTER a full decoder block. Hooks are removed even on failure.
"""
from contextlib import contextmanager
import torch


def hidden_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output, hidden):
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def changed_activation(hidden, vector, mode, alpha=1.0, center=None):
    result = hidden.clone()
    last = result[:, -1, :]
    vector = vector.to(device=last.device, dtype=last.dtype)
    if vector.shape != last.shape[-1:]:
        raise ValueError(f"Vector shape {vector.shape} does not match target {last.shape[-1:]}")
    if mode == "add":
        last.add_(alpha * vector)
    elif mode == "suppress":
        if center is None or center.shape != vector.shape:
            raise ValueError("Suppression requires the matching extraction baseline center.")
        norm = vector.float().norm()
        if norm <= 1e-10:
            raise ValueError("Cannot suppress a zero direction.")
        unit = vector.float() / norm
        offset = last.float() - center.to(last.device).float()
        projection = (offset * unit).sum(-1, keepdim=True)
        last.sub_((alpha * projection * unit).to(last.dtype))
    elif mode == "patch":
        last.copy_(vector.expand_as(last))
    else:
        raise ValueError(f"Unknown intervention {mode}")
    return result


class ActivationEngine:
    def __init__(self, model, tokenizer=None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError("This engine supports model.model.layers decoders; validate adapters for other architectures.")

    def target(self, layer, head=None):
        if not 0 <= layer < len(self.model.model.layers):
            raise ValueError("Layer outside model.")
        block = self.model.model.layers[layer]
        if head is None:
            return block, None
        attention = block.self_attn
        n_heads = self.model.config.num_attention_heads
        if not 0 <= head < n_heads:
            raise ValueError("Head outside model.")
        dim = attention.o_proj.in_features // n_heads
        if dim * n_heads != attention.o_proj.in_features:
            raise ValueError("Unexpected attention projection shape.")
        return attention.o_proj, slice(head * dim, (head + 1) * dim)

    def inputs(self, prompt):
        if self.tokenizer is None or not self.tokenizer.chat_template:
            raise ValueError("An explicit tokenizer chat template is required.")
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        return inputs.to(self.model.get_input_embeddings().weight.device)

    @torch.inference_mode()
    def capture(self, inputs, layer, head=None):
        module, section = self.target(layer, head)
        captured = []
        if section is None:
            def hook(_module, _args, output):
                captured.append(hidden_tensor(output)[:, -1, :].detach().float().cpu())
            handle = module.register_forward_hook(hook)
        else:
            def hook(_module, args):
                captured.append(args[0][:, -1, section].detach().float().cpu())
            handle = module.register_forward_pre_hook(hook)
        try:
            self.model(**inputs, use_cache=False)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape[0] != 1:
            raise ValueError("Extraction expects one unpadded prompt per forward pass.")
        return captured[0][0]

    @contextmanager
    def intervene(self, layer, vector, mode="add", alpha=1.0, center=None,
                  head=None, scope="each_step"):
        if scope not in ("each_step", "prefill"):
            raise ValueError("Scope must be each_step or prefill.")
        module, section = self.target(layer, head)
        stats = dict(forward_calls=0, interventions=0)

        def transform(hidden):
            active = scope == "each_step" or stats["forward_calls"] == 0
            stats["forward_calls"] += 1
            if not active:
                return hidden
            stats["interventions"] += 1
            if section is None:
                return changed_activation(hidden, vector, mode, alpha, center)
            output = hidden.clone()
            output[..., section] = changed_activation(hidden[..., section], vector, mode, alpha, center)
            return output

        if section is None:
            def hook(_module, _args, output):
                return replace_hidden(output, transform(hidden_tensor(output)))
            handle = module.register_forward_hook(hook)
        else:
            def hook(_module, args):
                return (transform(args[0]), *args[1:])
            handle = module.register_forward_pre_hook(hook)
        try:
            yield stats
        finally:
            handle.remove()

    @torch.inference_mode()
    def generate(self, prompt, seed=42, temperature=0.7, max_tokens=800,
                 layer=None, vector=None, mode="add", alpha=1.0,
                 center=None, head=None, scope="each_step"):
        inputs = self.inputs(prompt)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature > 0,
                      use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=1.0, top_k=0)
        stats = dict(forward_calls=0, interventions=0)
        if vector is None:
            tokens = self.model.generate(**inputs, **kwargs)
        else:
            with self.intervene(layer, vector, mode, alpha, center, head, scope) as stats:
                tokens = self.model.generate(**inputs, **kwargs)
        completion = tokens[0, inputs["input_ids"].shape[1]:]
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        finished = bool(completion.numel() and completion[-1].item() in eos_ids)
        return dict(Response=self.tokenizer.decode(completion, skip_special_tokens=True).strip(),
                    Generated_Tokens=completion.numel(), Finish_Reason="stop" if finished else "length",
                    Prompt_Tokens=inputs["input_ids"].shape[1],
                    Intervention_Calls=stats["interventions"], Forward_Calls=stats["forward_calls"])


def load_engine(model_id, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, device_map="auto", dtype="auto")
    return ActivationEngine(model, tokenizer)
