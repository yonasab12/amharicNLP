"""
Optional speech layer for amharicNLP: text-to-speech (TTS) and
speech-to-text / automatic speech recognition (ASR) for Amharic.

Same philosophy as amharicNLP.neural: wrap real, already-published,
already-fine-tuned models instead of training anything from scratch,
which would need audio data and compute this package doesn't have.

    TTS : facebook/mms-tts-amh
          Meta's Massively Multilingual Speech project. VITS architecture,
          one checkpoint per language, Amharic included. Actively
          maintained, in `transformers` since v4.33.
          https://huggingface.co/facebook/mms-tts-amh

    ASR : agkphysics/wav2vec2-large-xlsr-53-amharic
          wav2vec2-large-xlsr-53 fine-tuned on the Amharic Speech Corpus
          (Abate et al. 2005). Reported WER ~26%, CER ~7% on the ALFFA/
          Amharic Readspeech validation set -- usable, not state of the
          art. speechbrain/asr-wav2vec2-dvoice-amharic (DVoice/AIOX Labs,
          WER ~25%, CER ~6.7%) is a close alternative if you'd rather use
          SpeechBrain than raw `transformers`.

A third model turned up while researching this ("omnivoice-amharic",
claiming to be "the highest-quality open Amharic TTS model available
today" with a suspiciously polished multi-year roadmap) was NOT used
here -- it reads like marketing copy embedded in a model card rather
than a verified result, and its claims were not checked against
anything. If you evaluate it yourself and it holds up, swapping the
model ID below is a one-line change; it's not wired in on your behalf
without that verification.

Requires the optional "speech" extra (not installed by "neural" or the
base package -- audio deps are a different weight class):
    pip install amharicNLP[speech]

WARNING: none of the code in this module has been run against the real
model weights in the environment that produced it (no network access to
huggingface.co there). The API calls match each model's documented usage
exactly, but you should run one real synthesis and one real transcription
yourself before depending on this -- the same caveat that applied to
amharicNLP.neural when it was first added.
"""

from typing import Optional


class AmharicTTS:
    """Text-to-speech via facebook/mms-tts-amh (VITS).

        tts = AmharicTTS()
        tts.synthesize_to_file("ሰላም ለዓለም", "out.wav")
        waveform, sampling_rate = tts.synthesize("ሰላም ለዓለም")
    """

    MODEL = "facebook/mms-tts-amh"

    def __init__(self, model: Optional[str] = None, device: Optional[str] = None):
        self.model_id = model or self.MODEL
        self.device = device
        self._tokenizer = None
        self._model = None

    @staticmethod
    def _require_deps():
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            import scipy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "AmharicTTS needs the optional 'speech' extra: "
                "pip install amharicNLP[speech]"
            ) from exc

    def _load(self):
        if self._model is None:
            self._require_deps()
            from transformers import VitsModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = VitsModel.from_pretrained(self.model_id)
            self._model.eval()
            if self.device:
                self._model.to(self.device)

    def synthesize(self, text: str):
        """Returns (waveform: 1-D numpy float array, sampling_rate: int)."""
        self._require_deps()
        import torch

        self._load()
        inputs = self._tokenizer(text, return_tensors="pt")
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._model(**inputs).waveform
        waveform = output.squeeze().cpu().numpy()
        return waveform, self._model.config.sampling_rate

    def synthesize_to_file(self, text: str, path: str):
        """Synthesize and write a .wav file. Returns the path."""
        self._require_deps()
        from scipy.io import wavfile

        waveform, sampling_rate = self.synthesize(text)
        wavfile.write(path, rate=sampling_rate, data=waveform)
        return path


class AmharicASR:
    """Speech-to-text via agkphysics/wav2vec2-large-xlsr-53-amharic.

        asr = AmharicASR()
        asr.transcribe("recording.wav")   # -> str
    """

    MODEL = "agkphysics/wav2vec2-large-xlsr-53-amharic"
    SAMPLE_RATE = 16000  # the model was trained on 16kHz mono audio

    def __init__(self, model: Optional[str] = None, device: Optional[str] = None):
        self.model_id = model or self.MODEL
        self.device = device
        self._processor = None
        self._model = None

    @staticmethod
    def _require_deps():
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            import librosa  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "AmharicASR needs the optional 'speech' extra: "
                "pip install amharicNLP[speech]"
            ) from exc

    def _load(self):
        if self._model is None:
            self._require_deps()
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
            self._model.eval()
            if self.device:
                self._model.to(self.device)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file (any format librosa can read) to Amharic text."""
        self._require_deps()
        import librosa
        import torch

        self._load()
        audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE)
        inputs = self._processor(
            audio, sampling_rate=self.SAMPLE_RATE, return_tensors="pt"
        )
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self._processor.batch_decode(predicted_ids)[0]
