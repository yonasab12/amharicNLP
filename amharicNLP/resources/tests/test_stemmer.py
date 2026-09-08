from amharicNLP.resources.stemmer import AmharicStemmer, AmharicStemmerprocess


class TestAmharicStemmer:
    def test_negation_circumfix(self):
        # አል...ም negation: አልሄደም (did not go) -> ሄደ (root "go")
        assert AmharicStemmer().stemaize("አልሄደም") == "ሄደ"

    def test_progressive_prefix_and_person_suffix(self):
        # እየ...ች progressive + 3rd fem: እየሄደች -> ሄደ
        assert AmharicStemmer().stemaize("እየሄደች") == "ሄደ"

    def test_plural_suffix(self):
        assert AmharicStemmer().stemaize("ተማሪዎች") == "ተማሪ"

    def test_protected_words_are_not_touched(self):
        s = AmharicStemmer()
        for w in ["ሰላም", "ኢትዮጵያ", "ትምህርት"]:
            assert s.stemaize(w) == w

    def test_short_words_are_left_alone(self):
        assert AmharicStemmer().stemaize("እኔ") == "እኔ"

    def test_stemaize_never_swaps_in_an_unrelated_synonym(self):
        # Regression test: stemaize() must not fall back to the
        # related-term/synonym table when a word has no known affixes.
        assert AmharicStemmer().stemaize("ሄደ") == "ሄደ"

    def test_single_char_suffixes_do_not_chain_across_passes(self):
        # Regression test: with max_passes=2 (the old default), "ገበያው"
        # (the market) stripped suffix "ው" on pass 1 -> "ገበያ", then the
        # single-char suffix "ያ" matched again on pass 2 -> "ገበ". Two weak
        # single-character matches compounding did far more damage than
        # either alone. max_passes now defaults to 1.
        assert AmharicStemmer().stemaize("ገበያው") == "ገበያ"


class TestAmharicStemmerprocess:
    def test_stem_list_input(self):
        result = AmharicStemmerprocess().stem(["ተማሪዎች", "ኢትዮጵያ"])
        assert result == ["ተማሪ", "ኢትዮጵያ"]
