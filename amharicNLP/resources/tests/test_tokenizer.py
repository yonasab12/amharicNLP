from amharicNLP.resources.tokenizer import AmharicWordTokenizer


class TestAmharicWordTokenizer:
    def test_does_not_mangle_words_that_start_with_prefix_letters(self):
        # Regression test: an earlier version force-split any word
        # starting with የ/በ/ከ/ለ into two tokens, e.g. "የካቲት" -> "የ ካቲት".
        tok = AmharicWordTokenizer()
        assert tok.tokenize("የካቲት") == ["የካቲት"]
        assert tok.tokenize("በለስ") == ["በለስ"]

    def test_splits_punctuation(self):
        tok = AmharicWordTokenizer()
        tokens = tok.tokenize("ሰላም ነው።")
        assert tokens == ["ሰላም", "ነው", "።"]
