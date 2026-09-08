import pytest

amharic_tokenizer = pytest.importorskip(
    "amharic_tokenizer", reason="amharic-tokenizer not installed"
)

from amharicNLP.subword import AmharicSubwordTokenizer
from amharicNLP import AmharicNLP


class TestAmharicSubwordTokenizer:
    def test_tokenize_detokenize_roundtrip(self):
        sub = AmharicSubwordTokenizer()
        text = "የካቲት ወር ላይ ተማሪዎች ናቸው።"
        tokens = sub.tokenize(text)
        assert sub.detokenize(tokens) == text

    def test_token_count(self):
        sub = AmharicSubwordTokenizer()
        assert sub.token_count("ሰላም") > 0


class TestDocSubwordTokens:
    def test_doc_exposes_subword_tokens(self):
        nlp = AmharicNLP()
        doc = nlp("ሰላም ለዓለም")
        assert isinstance(doc.subword_tokens, list)
        assert len(doc.subword_tokens) > 0
