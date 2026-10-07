"""Compatibility API for local injection; no random dummy-vector demonstration."""
from interventions import load_engine


class ConceptVectorInjector:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct", use_ndif=False, revision=None):
        """Load a local activation engine for the compatibility injector API.

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

    def generate_with_intervention(self, prompt, concept_vector, layer_idx, alpha=1.0,
                                   target="residual", head_idx=None, seed=42):
        """Return a completion generated with additive steering at every forward step.

        Args:
            prompt: User-task text to format with the model's chat template.
            concept_vector: Direction in residual or pre-o_proj head coordinates.
            layer_idx: Zero-based target decoder block index.
            alpha: Signed multiplier applied to concept_vector.
            target: "residual" or "attention".
            head_idx: Required query-head index for attention; None for residual.
            seed: Generation sampling seed.
        Returns:
            Completion string, using the engine's default temperature/token limit.
        Raises:
            ValueError: For an unsupported target or inconsistent head_idx.
        Notes:
            Use activation_steer.py for saved experiments, controls, or scope selection.
        """
        if target not in ("residual", "attention"):
            raise ValueError("target must be residual or attention")
        if target == "attention" and head_idx is None:
            raise ValueError("Attention intervention requires an explicit head_idx.")
        if target == "residual" and head_idx is not None:
            raise ValueError("Residual intervention does not take a head_idx.")
        return self.engine.generate(prompt, seed=seed, layer=layer_idx, vector=concept_vector,
                                    alpha=alpha, head=head_idx)["Response"]


if __name__ == "__main__":
    raise SystemExit("Use: python scripts/activation_steer.py generate --help")
