"""Explicit local decoding settings shared by behavioral and intervention runners."""
import math


def sampling_kwargs(temperature=0.7, top_p=1.0, top_k=0):
    """Validate decoding settings and return Transformers generation overrides.

    Zero temperature uses greedy decoding. Its p/k values are neutralized because
    sampling filters do not apply; beam search and extra truncation are disabled.
    """
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0, 1]')
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer')
    return dict(do_sample=temperature > 0, temperature=temperature if temperature else 1.0,
                top_p=top_p if temperature else 1.0, top_k=top_k if temperature else 0,
                num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
                typical_p=1.0, min_p=None, epsilon_cutoff=0.0, eta_cutoff=0.0)
