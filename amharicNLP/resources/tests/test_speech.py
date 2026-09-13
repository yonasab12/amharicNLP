import pytest

from amharicNLP import AmharicTTS, AmharicASR


class TestSpeechModuleImportSafety:
    def test_importing_amharicNLP_does_not_require_speech_deps(self):
        # speech.py must have no module-level heavy imports (torch,
        # transformers, scipy, librosa) -- only inside methods -- so a
        # plain `pip install amharicNLP` (no [speech] extra) can still
        # import the package and construct these classes.
        tts = AmharicTTS()
        asr = AmharicASR()
        assert tts.model_id == "facebook/mms-tts-amh"
        assert asr.model_id == "agkphysics/wav2vec2-large-xlsr-53-amharic"


class TestSpeechDependencyGating:
    def test_tts_raises_clear_error_without_speech_extra(self):
        try:
            import transformers, torch, scipy  # noqa: F401
            pytest.skip("speech extra is installed; dependency-gating path not exercised")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="amharicNLP\\[speech\\]"):
            AmharicTTS().synthesize("ሰላም")

    def test_asr_raises_clear_error_without_speech_extra(self):
        try:
            import transformers, torch, librosa  # noqa: F401
            pytest.skip("speech extra is installed; dependency-gating path not exercised")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="amharicNLP\\[speech\\]"):
            AmharicASR().transcribe("nonexistent.wav")
