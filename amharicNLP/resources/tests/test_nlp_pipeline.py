import pytest

from amharicNLP import AmharicNLP


class TestAmharicNLPPipeline:
    def test_call_returns_doc_with_tokens_and_stems(self):
        nlp = AmharicNLP()
        doc = nlp("ተማሪዎች በትምህርት ላይ ናቸው።")
        assert doc.tokens == ["ተማሪዎች", "በትምህርት", "ላይ", "ናቸው", "።"]
        assert len(doc.stems) == len(doc.tokens)
        assert len(doc) == len(doc.tokens)
        assert list(doc) == doc.tokens

    def test_doc_has_pos_tags_aligned_with_tokens(self):
        nlp = AmharicNLP()
        doc = nlp("እኔ ኢትዮጵያዊ ነኝ።")
        assert [t for t, _ in doc.pos_tags] == doc.tokens
        assert doc.pos_tags[0] == ("እኔ", "PRON")
        assert doc.pos_tags[-1] == ("።", "PUNCT")

    def test_neural_attributes_raise_clear_error_without_neural_true(self):
        nlp = AmharicNLP()  # neural=False (default)
        doc = nlp("ጥሩ ነው።")
        with pytest.raises(RuntimeError):
            doc.sentiment
        with pytest.raises(RuntimeError):
            doc.entities
        with pytest.raises(RuntimeError):
            doc.embedding
