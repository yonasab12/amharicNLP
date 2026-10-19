from amharicNLP.resources.sentence_segmenter import AmharicSentenceSegmenter, Sentence


class TestAmharicSentenceSegmenter:
    def setup_method(self):
        self.seg = AmharicSentenceSegmenter()

    def test_basic_split_on_full_stop(self):
        assert self.seg.split("ዶ/ር አበበ መጣ። እሱ ጥሩ ነው።") == [
            "ዶ/ር አበበ መጣ።", "እሱ ጥሩ ነው።",
        ]

    def test_doubled_wordspace_full_stop_treated_as_one_boundary(self):
        assert self.seg.split("እሱ ደወለ፡፡ እሺ አልኩ፡፡") == [
            "እሱ ደወለ።", "እሺ አልኩ።",
        ]

    def test_question_and_exclamation_marks(self):
        assert self.seg.split("ናቸው? በጣም ጥሩ ነው!") == ["ናቸው?", "በጣም ጥሩ ነው!"]

    def test_does_not_split_on_comma_or_semicolon(self):
        result = self.seg.split("አንድ፣ ሁለት፣ ሶስት፤ አራት።")
        assert result == ["አንድ፣ ሁለት፣ ሶስት፤ አራት።"]

    def test_does_not_split_decimal_numbers(self):
        result = self.seg.split("ዋጋው 3.5 ብር ነው። ርካሽ ነው።")
        assert result == ["ዋጋው 3.5 ብር ነው።", "ርካሽ ነው።"]

    def test_does_not_split_inside_url(self):
        result = self.seg.split("ይመልከቱ www.example.com/path ላይ። አመሰግናለሁ።")
        assert result == ["ይመልከቱ www.example.com/path ላይ።", "አመሰግናለሁ።"]

    def test_does_not_split_known_abbreviation(self):
        result = self.seg.split("ዶ.ር አበበ ተናገረ። ጨርሷል።")
        assert result == ["ዶ.ር አበበ ተናገረ።", "ጨርሷል።"]

    def test_custom_abbreviations(self):
        seg = AmharicSentenceSegmenter(custom_abbreviations={"ልዩ.ቅ"})
        result = seg.split("ልዩ.ቅ ጉዳይ ነው። አለቀ።")
        assert result == ["ልዩ.ቅ ጉዳይ ነው።", "አለቀ።"]

    def test_empty_text_returns_empty_list(self):
        assert self.seg.split("") == []

    def test_no_terminator_returns_single_sentence(self):
        assert self.seg.split("ሰላም እንዴት ነህ") == ["ሰላም እንዴት ነህ"]

    def test_split_with_offsets_returns_sentence_namedtuples(self):
        result = self.seg.split_with_offsets("ሰላም። ደህና ነህ።")
        assert all(isinstance(s, Sentence) for s in result)
        assert [s.text for s in result] == ["ሰላም።", "ደህና ነህ።"]
        # Offsets index back into the original text.
        for s in result:
            assert s.text == "ሰላም። ደህና ነህ።"[s.start:s.end]

    def test_doubled_wordspace_offsets_index_original_text(self):
        text = "  እሱ ደወለ፡፡  እሺ አልኩ፡፡ "
        result = self.seg.split_with_offsets(text)
        assert [s.text for s in result] == ["እሱ ደወለ።", "እሺ አልኩ።"]
        assert [text[s.start:s.end] for s in result] == [
            "እሱ ደወለ፡፡", "እሺ አልኩ፡፡",
        ]

    def test_trims_whitespace_between_sentences(self):
        result = self.seg.split("ሰላም።    ደህና ነህ።")
        assert result == ["ሰላም።", "ደህና ነህ።"]

    def test_mixed_amharic_and_latin_punctuation(self):
        result = self.seg.split("This is English. ይሄ አማርኛ ነው።")
        assert result == ["This is English.", "ይሄ አማርኛ ነው።"]
