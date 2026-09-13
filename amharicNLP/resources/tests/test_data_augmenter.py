import random

from amharicNLP.utilities.data_augmenter import (
    augment_sentence,
    random_deletion,
    random_swap,
    synonym_replacement,
)


SENTENCE = "እንዴት ነህ ደስ ብሎኛል"


class TestAugmentSentence:
    def test_returns_a_string(self):
        # Regression test: augment_sentence() used to call
        # tokenizer.detokenize(), a method that doesn't exist on
        # AmharicWordTokenizer, so every call raised AttributeError.
        random.seed(0)
        result = augment_sentence(SENTENCE)
        assert isinstance(result, str)
        assert result != ""

    def test_is_deterministic_given_a_seed(self):
        random.seed(42)
        first = augment_sentence(SENTENCE)
        random.seed(42)
        second = augment_sentence(SENTENCE)
        assert first == second


class TestRandomDeletion:
    def test_short_input_is_untouched(self):
        assert random_deletion(["አንድ"]) == ["አንድ"]

    def test_never_returns_empty(self):
        random.seed(1)
        words = SENTENCE.split()
        result = random_deletion(words, p=1.0)
        assert len(result) >= 1


class TestSynonymReplacement:
    def test_replaces_known_synonym(self):
        random.seed(0)
        words = ["እንዴት", "ነህ"]
        result = synonym_replacement(words, n=2)
        assert result[0] in ["እንደምን", "በምን አይነት"]
        assert result[1] in ["ነሽ", "ናችሁ"]

    def test_leaves_unknown_words_alone(self):
        words = ["ኢትዮጵያ", "ውብ", "ናት"]
        result = synonym_replacement(words, n=1)
        assert result == words


class TestRandomSwap:
    def test_single_word_is_untouched(self):
        assert random_swap(["አንድ"]) == ["አንድ"]

    def test_swap_preserves_the_multiset_of_words(self):
        random.seed(2)
        words = SENTENCE.split()
        result = random_swap(words, n=2)
        assert sorted(result) == sorted(words)
