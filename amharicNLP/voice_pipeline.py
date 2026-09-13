"""
High-level voice AI pipeline: Audio -> STT -> Amharic NLP -> LLM ->
Amharic Text -> TTS -> Audio (spec section 12).

This module is the orchestration layer that connects the pieces the rest
of the package already provides -- it doesn't reimplement any of them:

    Audio -> AmharicASR (speech.py)        -> raw transcript
          -> AmharicNLP (nlp.py)           -> tokens/stems/pos/sentiment/etc.
          -> your LLM                       -> response text
          -> AmharicNormalizer (resources/) -> normalized response
          -> AmharicTTS (speech.py)        -> response audio

Per the package's core philosophy ("bring your own data ... connect it
to the AI model or architecture you choose"), this pipeline does NOT
hardcode or bundle an LLM -- there is no single obviously-right choice of
Amharic-capable LLM the way there is for ASR/TTS/sentiment/NER, and
tying this package to one would contradict the "not forcing developers
to use one particular LLM or training framework" goal from the dataset-
prep module. Instead, you inject a callable:

    def my_llm(transcript: str, doc) -> str:
        ...  # call whatever LLM/API you want, return Amharic response text

    pipeline = AmharicVoicePipeline(llm_fn=my_llm)
    result = pipeline.process_audio_turn("question.wav", output_audio_path="answer.wav")
    result.transcript       # what the user said (ASR output)
    result.doc               # Doc: tokens/pos_tags/sentiment_lexicon/... for the transcript
    result.response_text     # what the LLM said back (normalized)
    result.response_audio_path  # where the synthesized reply was written

`llm_fn` is optional: without it, `process_audio_turn`/`process_text_turn`
still run transcription + Amharic NLP understanding and raise a clear
error only if you try to go on to `respond`/synthesis, so you can use
this purely as an ASR->NLP pipeline (e.g. for a voice-driven analytics
tool with no conversational reply at all) without ever needing an LLM.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from .nlp import AmharicNLP, Doc
from .resources.normalizer import AmharicNormalizer


@dataclass
class VoiceTurnResult:
    transcript: Optional[str] = None
    doc: Optional[Doc] = None
    response_text: Optional[str] = None
    response_audio_path: Optional[str] = None
    response_waveform: Optional[object] = None
    response_sampling_rate: Optional[int] = None


class AmharicVoicePipeline:
    """
        pipeline = AmharicVoicePipeline(llm_fn=my_llm)

        # Full audio-in, audio-out turn (needs: pip install amharicNLP[speech]):
        result = pipeline.process_audio_turn("question.wav", output_audio_path="answer.wav")

        # Text-in, audio-out (skip ASR -- e.g. a chatbot with a TTS voice):
        result = pipeline.process_text_turn("ኢትዮጵያ ዋና ከተማ ማን ናት?", output_audio_path="answer.wav")

        # NLP-only, no LLM/TTS needed at all:
        doc = pipeline.understand("ይህ ጥሩ ፊልም ነበር።")
    """

    def __init__(
        self,
        asr=None,
        tts=None,
        nlp: Optional[AmharicNLP] = None,
        llm_fn: Optional[Callable[[str, Doc], str]] = None,
        device: Optional[str] = None,
    ):
        # ASR/TTS are constructed lazily-in-spirit: AmharicASR/AmharicTTS's
        # own __init__ does no model loading or dependency check, so
        # building this pipeline never requires the 'speech' extra unless
        # you actually call listen()/speak().
        if asr is None:
            from .speech import AmharicASR
            asr = AmharicASR(device=device)
        if tts is None:
            from .speech import AmharicTTS
            tts = AmharicTTS(device=device)
        self.asr = asr
        self.tts = tts
        self.nlp = nlp or AmharicNLP()
        self.llm_fn = llm_fn
        self._normalizer = AmharicNormalizer()

    # ---- Individual pipeline stages -------------------------------

    def listen(self, audio_path: str) -> str:
        """Audio -> transcript, via AmharicASR. Needs the 'speech' extra."""
        return self.asr.transcribe(audio_path)

    def understand(self, text: str) -> Doc:
        """Transcript/text -> Doc (tokens, stems, pos_tags, sentiment_lexicon,
        sentences, and -- if this pipeline's AmharicNLP was built with
        neural=True -- .sentiment/.entities/.embedding)."""
        return self.nlp(text)

    def respond(self, transcript: str, doc: Optional[Doc] = None) -> str:
        """transcript (+ Doc) -> response text, via the injected llm_fn.
        The response is normalized (AmharicNormalizer) before being
        returned, since it's about to be fed to TTS and/or shown as
        clean output text."""
        if self.llm_fn is None:
            raise RuntimeError(
                "AmharicVoicePipeline.respond() needs an llm_fn. Construct "
                "the pipeline with AmharicVoicePipeline(llm_fn=your_function), "
                "where your_function(transcript, doc) -> str."
            )
        if doc is None:
            doc = self.understand(transcript)
        raw_response = self.llm_fn(transcript, doc)
        return self._normalizer.normalize_text(raw_response)

    def speak(self, text: str, output_path: Optional[str] = None):
        """Response text -> audio, via AmharicTTS. Needs the 'speech'
        extra. If output_path is given, writes a .wav file and returns
        that path; otherwise returns (waveform, sampling_rate)."""
        if output_path:
            return self.tts.synthesize_to_file(text, output_path)
        return self.tts.synthesize(text)

    # ---- Full turns --------------------------------------------------

    def process_text_turn(self, text: str, output_audio_path: Optional[str] = None,
                            speak_response: bool = True) -> VoiceTurnResult:
        """Text -> NLP -> (LLM ->) (TTS ->) VoiceTurnResult. Skips ASR --
        for a text-chat-with-voice-reply use case, or for testing the
        NLP/LLM stages without audio input."""
        doc = self.understand(text)
        result = VoiceTurnResult(transcript=text, doc=doc)

        if self.llm_fn is not None:
            result.response_text = self.respond(text, doc)
            if speak_response:
                if output_audio_path:
                    result.response_audio_path = self.speak(result.response_text, output_audio_path)
                else:
                    waveform, sr = self.speak(result.response_text)
                    result.response_waveform = waveform
                    result.response_sampling_rate = sr
        return result

    def process_audio_turn(self, audio_path: str, output_audio_path: Optional[str] = None,
                             speak_response: bool = True) -> VoiceTurnResult:
        """Audio -> ASR -> NLP -> (LLM ->) (TTS ->) VoiceTurnResult. The
        full pipeline described at module level. Needs the 'speech' extra
        for listen()/speak(); NLP understanding works regardless."""
        transcript = self.listen(audio_path)
        result = self.process_text_turn(
            transcript, output_audio_path=output_audio_path, speak_response=speak_response,
        )
        result.transcript = transcript
        return result
