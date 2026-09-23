"""Compatibility API for local extraction. Use activation_steer.py for saved experiments."""
from interventions import load_engine


class ConceptVectorExtractor:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct", use_ndif=False, revision=None):
        if use_ndif:
            raise ValueError("This tested engine uses local PyTorch hooks; NDIF is not implemented.")
        self.engine = load_engine(model_id, revision)

    def extract_head_vector(self, creative_prompt, standard_prompt, layer_idx, head_idx):
        return self._difference(creative_prompt, standard_prompt, layer_idx, head_idx)

    def extract_residual_vector(self, creative_prompt, standard_prompt, layer_idx):
        return self._difference(creative_prompt, standard_prompt, layer_idx, None)

    def _difference(self, positive, negative, layer, head):
        return (self.engine.capture(self.engine.inputs(positive), layer, head)
                - self.engine.capture(self.engine.inputs(negative), layer, head))


if __name__ == "__main__":
    raise SystemExit("Use: python scripts/activation_steer.py extract --help")
