import os

import pytest

from amharicNLP.voice_pipeline import AmharicVoicePipeline, VoiceTurnResult


class FakeASR:
    def __init__(self, transcript="ኢትዮጵያ ዋና ከተማ ማን ናት?"):
        self.transcript = transcript
        self.calls = []

    def transcribe(self, path):
        self.calls.append(path)
        return self.transcript


class FakeTTS:
    def __init__(self):
        self.file_calls = []
        self.waveform_calls = []

    def synthesize_to_file(self, text, path):
        self.file_calls.append((text, path))
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def synthesize(self, text):
        self.waveform_calls.append(text)
        return ([0.0, 0.1, 0.2], 16000)


def fake_llm(transcript, doc):
    assert doc is not None
    return "አዲስ አበባ ናት።"


@pytest.fixture
def pipeline():
    return AmharicVoicePipeline(asr=FakeASR(), tts=FakeTTS(), llm_fn=fake_llm)


class TestConstruction:
    def test_builds_default_asr_tts_without_speech_deps(self):
        # AmharicASR/AmharicTTS __init__ does no dependency check --
        # constructing the pipeline should never require the 'speech' extra.
        pipeline = AmharicVoicePipeline(llm_fn=lambda t, d: "ok")
        assert pipeline.asr is not None
        assert pipeline.tts is not None

    def test_accepts_injected_asr_tts(self):
        asr, tts = FakeASR(), FakeTTS()
        pipeline = AmharicVoicePipeline(asr=asr, tts=tts)
        assert pipeline.asr is asr
        assert pipeline.tts is tts

    def test_llm_fn_optional(self):
        pipeline = AmharicVoicePipeline(asr=FakeASR(), tts=FakeTTS())
        assert pipeline.llm_fn is None


class TestUnderstandOnly:
    def test_understand_returns_doc_with_tokens(self, pipeline):
        doc = pipeline.understand("ይህ ጥሩ ፊልም ነበር።")
        assert doc.tokens == ["ይህ", "ጥሩ", "ፊልም", "ነበር", "።"]

    def test_understand_works_without_llm_fn(self):
        pipeline = AmharicVoicePipeline(asr=FakeASR(), tts=FakeTTS())
        doc = pipeline.understand("ሰላም ለዓለም።")
        assert doc.tokens == ["ሰላም", "ለአለም", "።"]


class TestRespond:
    def test_respond_raises_without_llm_fn(self):
        pipeline = AmharicVoicePipeline(asr=FakeASR(), tts=FakeTTS())
        with pytest.raises(RuntimeError, match="llm_fn"):
            pipeline.respond("ሰላም")

    def test_respond_normalizes_llm_output(self, pipeline):
        response = pipeline.respond("ኢትዮጵያ ዋና ከተማ ማን ናት?")
        assert response == "አዲስ አበባ ናት ።"

    def test_respond_passes_transcript_and_doc_to_llm_fn(self):
        seen = {}

        def capturing_llm(transcript, doc):
            seen["transcript"] = transcript
            seen["tokens"] = doc.tokens
            return "ምላሽ"

        pipeline = AmharicVoicePipeline(asr=FakeASR(), tts=FakeTTS(), llm_fn=capturing_llm)
        pipeline.respond("ሰላም ለዓለም")
        assert seen["transcript"] == "ሰላም ለዓለም"
        assert "ሰላም" in seen["tokens"]


class TestListenAndSpeak:
    def test_listen_delegates_to_asr(self, pipeline):
        transcript = pipeline.listen("some.wav")
        assert transcript == "ኢትዮጵያ ዋና ከተማ ማን ናት?"
        assert pipeline.asr.calls == ["some.wav"]

    def test_speak_without_path_returns_waveform(self, pipeline):
        waveform, sr = pipeline.speak("ሰላም")
        assert waveform == [0.0, 0.1, 0.2]
        assert sr == 16000

    def test_speak_with_path_writes_file(self, pipeline, tmp_path):
        out = str(tmp_path / "out.wav")
        result_path = pipeline.speak("ሰላም", output_path=out)
        assert result_path == out
        assert os.path.exists(out)


class TestProcessTextTurn:
    def test_full_text_turn_with_file_output(self, pipeline, tmp_path):
        out = str(tmp_path / "reply.wav")
        result = pipeline.process_text_turn("ኢትዮጵያ ዋና ከተማ ማን ናት?", output_audio_path=out)
        assert isinstance(result, VoiceTurnResult)
        assert result.transcript == "ኢትዮጵያ ዋና ከተማ ማን ናት?"
        assert result.response_text == "አዲስ አበባ ናት ።"
        assert result.response_audio_path == out
        assert os.path.exists(out)

    def test_text_turn_without_file_returns_waveform(self, pipeline):
        result = pipeline.process_text_turn("ሰላም")
        assert result.response_waveform == [0.0, 0.1, 0.2]
        assert result.response_sampling_rate == 16000
        assert result.response_audio_path is None

    def test_text_turn_skips_llm_when_not_configured(self):
        pipeline = AmharicVoicePipeline(asr=FakeASR(), tts=FakeTTS())
        result = pipeline.process_text_turn("ሰላም")
        assert result.response_text is None
        assert result.response_audio_path is None
        assert result.doc is not None  # NLP understanding still ran

    def test_speak_response_false_skips_tts(self):
        tts = FakeTTS()
        pipeline = AmharicVoicePipeline(asr=FakeASR(), tts=tts, llm_fn=fake_llm)
        result = pipeline.process_text_turn("ሰላም", speak_response=False)
        assert result.response_text == "አዲስ አበባ ናት ።"
        assert tts.file_calls == []
        assert tts.waveform_calls == []


class TestProcessAudioTurn:
    def test_full_audio_turn(self, pipeline, tmp_path):
        out = str(tmp_path / "reply.wav")
        result = pipeline.process_audio_turn("question.wav", output_audio_path=out)
        assert result.transcript == "ኢትዮጵያ ዋና ከተማ ማን ናት?"
        assert pipeline.asr.calls == ["question.wav"]
        assert result.response_text == "አዲስ አበባ ናት ።"
        assert os.path.exists(out)

    def test_audio_turn_doc_reflects_transcript(self, pipeline):
        result = pipeline.process_audio_turn("question.wav", speak_response=False)
        assert result.doc.tokens[0] == "ኢትዮጵያ"
