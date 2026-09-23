"""Compatibility API for local injection; no random dummy-vector demonstration."""
from interventions import load_engine


class ConceptVectorInjector:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct", use_ndif=False, revision=None):
        if use_ndif:
            raise ValueError("This tested engine uses local PyTorch hooks; NDIF is not implemented.")
        self.engine = load_engine(model_id, revision)

    def generate_with_intervention(self, prompt, concept_vector, layer_idx, alpha=1.0,
                                   target="residual", head_idx=None, seed=42):
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
