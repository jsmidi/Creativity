"""Extract response-level attention-head activations from saved generations."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ActivationEngine:
    """Teacher-force saved prompt/response pairs and mean-pool attention heads."""

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", dtype="auto", local_files_only=True
        ).eval()

    def _prompt_ids(self, prompt, prompt_token_ids=None):
        device = self.model.get_input_embeddings().weight.device
        if prompt_token_ids is not None:
            ids = torch.as_tensor(prompt_token_ids, dtype=torch.long, device=device).reshape(1, -1)
            if ids.shape[1] == 0:
                raise ValueError("Empty prompt token sequence.")
            return ids
        if self.tokenizer.chat_template:
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs["input_ids"].to(device)

    def _response_ids_without_eos(self, response, response_token_ids=None):
        if response_token_ids is None:
            response_ids = self.tokenizer(response, add_special_tokens=False,
                                          return_tensors="pt")["input_ids"][0]
        else:
            response_ids = torch.as_tensor(response_token_ids, dtype=torch.long).reshape(-1)

        eos_ids = set()
        generation_config = getattr(self.model, "generation_config", None)
        configured_eos = getattr(generation_config, "eos_token_id", None)
        if configured_eos is not None:
            configured_eos = (configured_eos if isinstance(configured_eos, (list, tuple, set))
                              else [configured_eos])
            eos_ids.update(int(token_id) for token_id in configured_eos)
        tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
        if tokenizer_eos is not None:
            eos_ids.add(int(tokenizer_eos))

        # Stop before the first configured end token. Model generation configs
        # may define several end tokens, while tokenizer.eos_token_id is scalar.
        if eos_ids:
            for index, token_id in enumerate(response_ids.tolist()):
                if token_id in eos_ids:
                    return response_ids[:index]
        return response_ids

    def _layers_and_head_count(self):
        layers = None
        for path in (("model", "layers"), ("model", "language_model", "layers"),
                     ("language_model", "layers")):
            module = self.model
            for attribute in path:
                module = getattr(module, attribute, None)
                if module is None:
                    break
            if module is not None and len(module) and all(
                    hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj")
                    for block in module):
                layers = module
                break
        if layers is None:
            raise ValueError("Model does not expose decoder layers with self_attn.o_proj hooks.")
        text_config = getattr(self.model.config, "text_config", None)
        n_heads = getattr(self.model.config, "num_attention_heads", None)
        if n_heads is None and text_config is not None:
            n_heads = getattr(text_config, "num_attention_heads", None)
        if n_heads is None:
            raise ValueError("Model config does not expose num_attention_heads.")
        return layers, int(n_heads)

    @torch.inference_mode()
    def prompt_response_head_means(self, prompt, response, response_token_ids=None,
                                   positions=("response", "last-prompt"), prompt_token_ids=None):
        """Capture selected prompt/response representations in one forward pass.

        Prefer saved formatted prompt IDs and exact generated response IDs.
        Older CSVs can reconstruct either sequence from its saved text.
        Returns a mapping from requested position names to CPU float32 tensors
        shaped [layers, heads, head_dim].
        """
        allowed = {"response", "last-prompt", "response-predictor"}
        positions = tuple(dict.fromkeys(positions))
        if not positions or not set(positions) <= allowed:
            raise ValueError(f"positions must be selected from {sorted(allowed)}")
        prompt_ids = self._prompt_ids(prompt, prompt_token_ids)
        response_ids = self._response_ids_without_eos(response, response_token_ids)
        if response_ids.numel() == 0:
            raise ValueError("Empty response token sequence.")

        layers, n_heads = self._layers_and_head_count()
        prompt_length = prompt_ids.shape[1]
        full_ids = torch.cat((prompt_ids, response_ids.unsqueeze(0).to(prompt_ids.device)), dim=1)
        captured, handles = {}, []

        def make_hook(layer_index):
            def capture(_module, projection_inputs):
                values = projection_inputs[0]
                if values.shape[0] != 1:
                    raise ValueError("Activation extraction expects one unpadded sequence.")
                layer_values = {}
                if "last-prompt" in positions:
                    layer_values["last-prompt"] = values[0, prompt_length - 1].detach().float().cpu().reshape(n_heads, -1)
                if "response" in positions:
                    answer = values[0, prompt_length:prompt_length + response_ids.numel()]
                    if answer.shape[0] == 0:
                        raise ValueError("No answer-token positions reached the attention output hook.")
                    layer_values["response"] = answer.detach().float().mean(0).cpu().reshape(n_heads, -1)
                if "response-predictor" in positions:
                    predictors = values[0, prompt_length - 1:
                                         prompt_length + response_ids.numel() - 1]
                    if predictors.shape[0] != response_ids.numel():
                        raise ValueError("Could not align one predictor state to each response token.")
                    layer_values["response-predictor"] = (
                        predictors.detach().float().mean(0).cpu().reshape(n_heads, -1))
                captured[layer_index] = layer_values
            return capture

        try:
            for index, block in enumerate(layers):
                handles.append(block.self_attn.o_proj.register_forward_pre_hook(make_hook(index)))
            self.model(input_ids=full_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != len(layers):
            raise RuntimeError("Failed to capture all transformer layers.")
        return {position: torch.stack([captured[index][position] for index in range(len(layers))])
                for position in positions}

    @torch.inference_mode()
    def prompt_response_residual_means(self, prompt, response, response_token_ids=None,
                                       positions=("response", "last-prompt"), prompt_token_ids=None):
        """Capture post-block residual states for selected prompt/response positions.

        Returns CPU float32 tensors shaped [layers, hidden_size]. The states are
        captured after each decoder block and before any final model norm.
        """
        allowed = {"response", "last-prompt", "response-predictor"}
        positions = tuple(dict.fromkeys(positions))
        if not positions or not set(positions) <= allowed:
            raise ValueError(f"positions must be selected from {sorted(allowed)}")
        prompt_ids = self._prompt_ids(prompt, prompt_token_ids)
        response_ids = self._response_ids_without_eos(response, response_token_ids)
        if ({"response", "response-predictor"} & set(positions)
                and response_ids.numel() == 0):
            raise ValueError("Empty response token sequence.")

        layers, _ = self._layers_and_head_count()
        prompt_length = prompt_ids.shape[1]
        full_ids = torch.cat((prompt_ids, response_ids.unsqueeze(0).to(prompt_ids.device)), dim=1)
        captured, handles = {}, []

        def make_hook(layer_index):
            def capture(_module, _inputs, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError("Expected one unpadded [batch, tokens, hidden] decoder output.")
                layer_values = {}
                if "last-prompt" in positions:
                    layer_values["last-prompt"] = hidden[0, prompt_length - 1].detach().float().cpu()
                if "response" in positions:
                    answer = hidden[0, prompt_length:prompt_length + response_ids.numel()]
                    if answer.shape[0] == 0:
                        raise ValueError("No response-token positions reached the decoder block.")
                    layer_values["response"] = answer.detach().float().mean(0).cpu()
                if "response-predictor" in positions:
                    predictors = hidden[0, prompt_length - 1:
                                         prompt_length + response_ids.numel() - 1]
                    if predictors.shape[0] != response_ids.numel():
                        raise ValueError("Could not align one predictor state to each response token.")
                    layer_values["response-predictor"] = predictors.detach().float().mean(0).cpu()
                captured[layer_index] = layer_values
            return capture

        try:
            for index, block in enumerate(layers):
                handles.append(block.register_forward_hook(make_hook(index)))
            self.model(input_ids=full_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != len(layers):
            raise RuntimeError("Failed to capture all transformer layers.")
        return {position: torch.stack([captured[index][position] for index in range(len(layers))])
                for position in positions}
