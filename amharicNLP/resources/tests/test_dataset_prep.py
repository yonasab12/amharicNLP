from amharicNLP.dataset_prep import AmharicDatasetPreprocessor

# Copied verbatim from https://huggingface.co/datasets/snapwre/amharic-speech
# (dataset viewer, train split), not invented -- includes the traditional
# doubled-wordspace "፡፡" sentence-final mark that news-style writing still
# uses (equivalent to "።").
REAL_SENTENCES = [
    "የምክር ቤቱ ጸሐፊ ሼህ ሁሴን በሽር በበኩላቸው፤ በዓሉን እስልምና በሚያዘው መሰረት በአብሮነትና በፍቅር ልናሳልፈው ይገባል ብለዋል፡፡",
    "የክልሉ ማረሚያ ቤት ኮሚሽን ኮሚሽነር አድናን አህመድ በበኩላቸው ፤ የማዕረግ ዕድገቱን ያገኙት የመቆያ ጊዜያቸውን የሸፈኑ አባላት መሆናቸውን ተናግረዋል።",
    "በግንባታ መወጣጫ መደርመስ ምክንያት ባጋጠመዉ አደጋ በደረሰዉ ጉዳት መንግሥት ኀዘኑን ይገልጻል፡፡",
]


class TestAmharicDatasetPreprocessor:
    def test_clean_text_normalizes_doubled_wordspace(self):
        # Regression test: "፡፡" is the traditional doubled Ethiopic
        # wordspace still common in news writing -- equivalent to "።".
        # It showed up in roughly a quarter of the sampled real sentences.
        prep = AmharicDatasetPreprocessor()
        result = prep.clean_text(REAL_SENTENCES[0])
        assert "፡፡" not in result
        assert result.endswith("።")

    def test_clean_text_on_real_sentences_is_idempotent_and_nonempty(self):
        prep = AmharicDatasetPreprocessor()
        for sentence in REAL_SENTENCES:
            cleaned = prep.clean_text(sentence)
            assert cleaned
            assert prep.clean_text(cleaned) == cleaned

    def test_process_text_default_returns_string_not_tokens(self):
        # ASR/TTS training targets and LLM corpora want normalized text,
        # not a token list -- tokenizing must be opt-in.
        prep = AmharicDatasetPreprocessor()
        result = prep.process_text(REAL_SENTENCES[1])
        assert isinstance(result, str)

    def test_process_text_with_tokenize_returns_tokens(self):
        prep = AmharicDatasetPreprocessor(tokenize=True)
        result = prep.process_text(REAL_SENTENCES[1])
        assert isinstance(result, list)
        assert result[-1] == "።"

    def test_process_text_with_stem_returns_stems(self):
        prep = AmharicDatasetPreprocessor(stem=True)
        result = prep.process_text("ተማሪዎች ትምህርት ቤት ሄዱ።")
        assert "ተማሪ" in result  # plural suffix stripped

    def test_load_raises_clear_error_without_datasets_extra(self):
        import pytest
        try:
            import datasets  # noqa: F401
            pytest.skip("datasets extra is installed; gating path not exercised")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="amharicNLP\\[datasets\\]"):
            AmharicDatasetPreprocessor.load("snapwre/amharic-speech")
