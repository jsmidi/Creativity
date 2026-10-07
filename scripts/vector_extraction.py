"""Compatibility API for local extraction. Use activation_steer.py for saved experiments."""
from interventions import load_engine


class ConceptVectorExtractor:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct", use_ndif=False, revision=None):
        """Load a local activation engine for the compatibility extractor API.

        Args:
            model_id: Local checkpoint directory or Hugging Face model identifier.
            use_ndif: Legacy flag; must be False because remote NDIF is unsupported.
            revision: Optional Hugging Face revision.
        Raises:
            ValueError: If use_ndif is True. Model loading can download remote weights.
        """
        if use_ndif:
            raise ValueError("This tested engine uses local PyTorch hooks; NDIF is not implemented.")
        self.engine = load_engine(model_id, revision)

    def extract_head_vector(self, creative_prompt, standard_prompt, layer_idx, head_idx):
        """Compute a single prompt pair's Creative-minus-Standard head contrast.

        Args:
            creative_prompt: Positive-condition user text, before chat formatting.
            standard_prompt: Baseline-condition user text, before chat formatting.
            layer_idx: Zero-based decoder block index.
            head_idx: Zero-based query-head index before o_proj.
        Returns:
            CPU float32 tensor [head_dim]. No across-example averaging or file saving.
        """
        return self._difference(creative_prompt, standard_prompt, layer_idx, head_idx)

    def extract_residual_vector(self, creative_prompt, standard_prompt, layer_idx):
        """Compute a single prompt pair's post-block residual contrast.

        Args:
            creative_prompt: Positive-condition user text.
            standard_prompt: Baseline-condition user text.
            layer_idx: Zero-based decoder block index.
        Returns:
            CPU float32 tensor [hidden_size] for the last formatted prompt token.
        """
        return self._difference(creative_prompt, standard_prompt, layer_idx, None)

    def _difference(self, positive, negative, layer, head):
        """Capture two prompt states and subtract the negative state from the positive.

        Args:
            positive: Positive-condition user prompt.
            negative: Baseline-condition user prompt.
            layer: Zero-based decoder block index.
            head: Query-head index, or None for a residual contrast.
        Returns:
            Detached CPU float32 difference vector.
        """
        return (self.engine.capture(self.engine.inputs(positive), layer, head)
                - self.engine.capture(self.engine.inputs(negative), layer, head))


if __name__ == "__main__":
    raise SystemExit("Use: python scripts/activation_steer.py extract --help")
