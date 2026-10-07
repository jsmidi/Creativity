"""Local causal interventions for Llama-style decoders, including true head slices.

Head coordinates are BEFORE o_proj, not slices of its mixed output. Residual
coordinates are AFTER a full decoder block. Hooks are removed even on failure.
"""
from contextlib import contextmanager
from sampling import sampling_kwargs
import torch


def hidden_tensor(output):
    """Return the activation tensor from a decoder block result.

    Args:
        output: A hidden-state tensor, or a tuple whose first entry is that tensor.
    Returns:
        The original tensor, without copying or detaching it.
    """
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output, hidden):
    """Replace hidden states while preserving a block's output structure.

    Args:
        output: Original tensor or tuple returned by the block.
        hidden: Replacement hidden-state tensor of the expected shape.
    Returns:
        hidden directly, or a tuple containing hidden followed by unchanged entries.
    """
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def changed_activation(hidden, vector, mode, alpha=1.0, center=None):
    """Modify only the final sequence position in a cloned activation tensor.

    Args:
        hidden: Tensor shaped [batch, sequence, features]; remains unchanged.
        vector: One-dimensional direction, or donor state for patch mode.
        mode: "add", "suppress", or "patch".
        alpha: Signed addition strength or suppression fraction; ignored by patch.
        center: Baseline vector required for centered projection suppression.
    Returns:
        Modified tensor with the same shape, dtype, and device as hidden.
    Raises:
        ValueError: For incompatible shapes, unsupported modes, or a zero
            suppression direction. Suppression removes alpha times the coordinate
            of hidden - center along the unit direction; patch replaces the state.
    """
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
        """Wrap an evaluation-mode decoder for local capture and intervention.

        Args:
            model: Loaded causal language model exposing model.layers; Llama is tested.
            tokenizer: Optional chat tokenizer, required for prompt-based generation.
        Raises:
            ValueError: If the decoder does not expose the expected layer structure.
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError("This engine supports model.model.layers decoders; validate adapters for other architectures.")

    def target(self, layer, head=None):
        """Resolve the module and coordinate slice for an intervention.

        Args:
            layer: Zero-based decoder block index.
            head: Zero-based query-head index, or None for a full post-block residual.
        Returns:
            (module, slice): the block and None for residuals, or o_proj and its
            input-coordinate slice for a head. These are not KV-head indices.
        Raises:
            ValueError: For invalid indices or an incompatible projection shape.
        """
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
        """Format one user prompt with the chat template and move tokens to the model.

        Args:
            prompt: Unformatted user-task text.
        Returns:
            Tokenizer batch with input_ids and attention_mask on the embedding device.
        Raises:
            ValueError: If no tokenizer with a chat template is available.
        """
        if self.tokenizer is None or not self.tokenizer.chat_template:
            raise ValueError("An explicit tokenizer chat template is required.")
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        return inputs.to(self.model.get_input_embeddings().weight.device)

    @torch.inference_mode()
    def capture(self, inputs, layer, head=None):
        """Capture the last prompt token at one residual or attention-head target.

        Args:
            inputs: Tokenized model inputs for one unpadded prompt.
            layer: Zero-based decoder block index.
            head: Query-head index before o_proj, or None for the post-block residual.
        Returns:
            Detached CPU float32 vector: [hidden_size] or [head_dim].
        Notes:
            Runs one forward pass without a KV cache or gradients. The temporary hook
            is removed even if the forward pass fails; no answer is generated.
        """
        module, section = self.target(layer, head)
        captured = []
        if section is None:
            def hook(_module, _args, output):
                """Capture or replace activations for the enclosing hook operation.

                Args:
                    _module: Module invoking the hook; unused.
                    _args: Positional block inputs; unused.
                    output: Original decoder block tensor or tuple output.
                Returns:
                    None for capture hooks; a replacement output/input tuple for edit hooks.
                    Uses the target and storage closed over by the enclosing function.
                """
                captured.append(hidden_tensor(output)[:, -1, :].detach().float().cpu())
            handle = module.register_forward_hook(hook)
        else:
            def hook(_module, args):
                """Capture or replace activations for the enclosing hook operation.

                Args:
                    _module: Module invoking the hook; unused.
                    args: Projection positional inputs; args[0] is the activation tensor.
                Returns:
                    None for capture hooks; a replacement output/input tuple for edit hooks.
                    Uses the target and storage closed over by the enclosing function.
                """
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
        """Temporarily install a hook that edits the target's last-token activation.

        Args:
            layer: Zero-based block index.
            vector: Residual/head direction, or donor activation for patch mode.
            mode: "add", "suppress", or "patch"; see changed_activation.
            alpha: Addition multiplier or suppression fraction; ignored by patch.
            center: Matching baseline center for suppression; otherwise unused.
            head: Query-head index, or None for the full residual.
            scope: "prefill" edits only the first target forward call; "each_step"
                edits every call, including prefill and cached decoding steps.
        Yields:
            Mutable dictionary counting forward_calls and interventions.
        Notes:
            Used as a context manager; removes the hook on normal or exceptional exit.
            It does not change model weights or edit earlier sequence positions.
        """
        if scope not in ("each_step", "prefill"):
            raise ValueError("Scope must be each_step or prefill.")
        module, section = self.target(layer, head)
        stats = dict(forward_calls=0, interventions=0)

        def transform(hidden):
            """Apply the scoped intervention and update shared hook counters.

            Args:
                hidden: Current residual tensor or concatenated head-output tensor.
            Returns:
                Unchanged tensor if the scope is inactive, otherwise a modified clone.
                The enclosing layer/head selection determines which coordinates change.
            """
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
                """Capture or replace activations for the enclosing hook operation.

                Args:
                    _module: Module invoking the hook; unused.
                    _args: Positional block inputs; unused.
                    output: Original decoder block tensor or tuple output.
                Returns:
                    None for capture hooks; a replacement output/input tuple for edit hooks.
                    Uses the target and storage closed over by the enclosing function.
                """
                return replace_hidden(output, transform(hidden_tensor(output)))
            handle = module.register_forward_hook(hook)
        else:
            def hook(_module, args):
                """Capture or replace activations for the enclosing hook operation.

                Args:
                    _module: Module invoking the hook; unused.
                    args: Projection positional inputs; args[0] is the activation tensor.
                Returns:
                    None for capture hooks; a replacement output/input tuple for edit hooks.
                    Uses the target and storage closed over by the enclosing function.
                """
                return (transform(args[0]), *args[1:])
            handle = module.register_forward_pre_hook(hook)
        try:
            yield stats
        finally:
            handle.remove()

    @torch.inference_mode()
    def generate(self, prompt, seed=42, temperature=0.7, max_tokens=800,
                 layer=None, vector=None, mode="add", alpha=1.0,
                 center=None, head=None, scope="each_step", top_p=1.0, top_k=0):
        """Generate one completion, optionally with a temporary activation intervention.

        Args:
            prompt: User-task text to format with the tokenizer's chat template.
            seed: PyTorch CPU/CUDA sampling seed, reset for this generation.
            temperature: Sampling temperature; zero selects greedy decoding.
            max_tokens: Maximum number of new tokens, excluding the prompt.
            layer: Target block index when vector is supplied.
            vector: Direction/donor tensor; None generates without intervention.
            mode: "add", "suppress", or "patch" when vector is supplied.
            alpha: Intervention strength; ignored by patch.
            center: Baseline center required for suppression.
            head: Query-head index, or None for a residual intervention.
            scope: "prefill" or "each_step"; passed to intervene.
            top_p: Nucleus probability in (0, 1], used only when sampling.
            top_k: Nonnegative candidate limit; zero disables this filter.
        Returns:
            Dictionary containing Response, prompt/completion token counts,
            Finish_Reason, and hook counters. Only completion tokens are decoded.
        Notes:
            top_p/top_k default to 1/0 (no truncation). Seeds do not guarantee reproducibility
            across different devices or kernels.
        """
        inputs = self.inputs(prompt)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature > 0,
                      use_cache=True, pad_token_id=self.tokenizer.eos_token_id)
        kwargs.update(sampling_kwargs(temperature, top_p, top_k))
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
    """Load a Transformers model/tokenizer and wrap them in ActivationEngine.

    Args:
        model_id: Local model directory or Hugging Face repository identifier.
        revision: Optional repository revision; None uses the loader default.
    Returns:
        ActivationEngine with automatic device placement and model dtype.
    Notes:
        Remote identifiers can trigger downloads. This is the Llama-style
        intervention loader, not the separate GPT-OSS behavioral loader.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, device_map="auto", dtype="auto")
    return ActivationEngine(model, tokenizer)
