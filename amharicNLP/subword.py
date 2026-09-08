"""
Subword (BPE) tokenizer for Amharic -- for feeding text into an LLM,
computing token counts, or training an embedding model.

This wraps the `amharic-tokenizer` PyPI package (Sefineh Tesfa,
https://github.com/sefineh-ai/Amharic-Tokenizer): a syllable-aware BPE
tokenizer over decomposed fidel with a Cython core and a pretrained model
included. It is NOT a replacement for AmharicWordTokenizer in
resources/tokenizer.py -- they solve different problems:

  - AmharicWordTokenizer (resources/tokenizer.py): splits text into whole
    words. That's what AmharicStemmer, AmharicStopwordProcessor, and the
    rest of the rule-based pipeline need to match against -- affix rules
    and stopword lists operate on whole words, not BPE fragments.

  - AmharicSubwordTokenizer (this module): splits words into BPE
    subword pieces with a fixed vocabulary, e.g. "የካቲት" ->
    ['የአከኣተኢተእ<eow>', ...]. That's what you feed into a transformer's
    embedding layer, or use to count tokens for an LLM context window.
    It is not meant to be read as words, and stemming/stopword-removal
    will not work on its output.

Use whichever matches what you're building: classic NLP pipeline
(stemming, stopwords, POS-style tasks) -> AmharicWordTokenizer.
Training/serving a language model -> AmharicSubwordTokenizer.
"""

from functools import lru_cache

DEFAULT_MODEL = "amh_bpe_v0.2.6"


class MissingSubwordDependencyError(ImportError):
    pass


@lru_cache(maxsize=4)
def _load(model_name: str):
    try:
        from amharic_tokenizer import AmharicTokenizer as _AmharicTokenizer
    except ImportError as exc:
        raise MissingSubwordDependencyError(
            "AmharicSubwordTokenizer needs the 'amharic-tokenizer' package: "
            "pip install amharic-tokenizer"
        ) from exc
    return _AmharicTokenizer.load(model_name)


class AmharicSubwordTokenizer:
    """
    Thin wrapper around the pretrained BPE model from `amharic-tokenizer`.
    Lazy: the model isn't loaded until you call tokenize/encode/decode.

        sub = AmharicSubwordTokenizer()
        sub.tokenize("የካቲት ወር ላይ ተማሪዎች ናቸው።")
        sub.encode("የካቲት ወር ላይ ተማሪዎች ናቸው።")     # -> list[int]
        sub.decode(ids)                            # -> str

    NOTE: tokenize() -> detokenize() round-trips exactly. encode() ->
    decode() does not always (trailing punctuation can be dropped) --
    that's a quirk in the underlying package, not this wrapper. Prefer
    tokenize()/detokenize() if you need an exact round trip.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

    @property
    def _tok(self):
        return _load(self.model_name)

    def tokenize(self, text: str):
        return self._tok.tokenize(text)

    def encode(self, text: str):
        return self._tok.encode(text)

    def decode(self, ids):
        return self._tok.decode(ids)

    def detokenize(self, tokens):
        return self._tok.detokenize(tokens)

    def token_count(self, text: str) -> int:
        """Handy for estimating LLM context-window usage."""
        return len(self.tokenize(text))
