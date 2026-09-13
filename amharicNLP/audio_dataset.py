"""
Audio dataset preparation shared by the STT (speech-to-text) and TTS
(text-to-speech) sides of the package -- section 10/11 of the spec.

`speech.py` wraps pretrained ASR/TTS *models*. This module is the data
layer in front of them: turning a raw pile of (audio, transcript) pairs
into something a Whisper-style, CTC, or encoder-decoder speech model (or
a TTS model, symmetrically) can actually train on -- duration filtering,
silence trimming, resampling, basic audio-quality checks, transcript
normalization (reusing AmharicNormalizer/AmharicSentenceSegmenter, not a
separate implementation), and train/validation/test splitting with
optional speaker-grouping so the same voice doesn't leak across splits.

Records are plain `dict`s, same convention as corpus.py/llm_dataset.py:
    {"audio_path": "clip001.wav", "text": "ሰላም ለዓለም", "speaker_id": "spk1", ...}
Any other metadata (duration_s, sample_rate, ...) passes through
untouched.

Dependency split, same pattern as speech.py:
  - Metadata-only operations (filtering by an already-known duration,
    train/val/test splitting, transcript normalization) are pure Python
    + stdlib, no extra install needed.
  - Anything that touches an actual audio file (computing duration from
    the file, trimming silence, resampling) needs the optional "speech"
    extra (librosa, soundfile): pip install amharicNLP[speech]
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .resources.normalizer import AmharicNormalizer
from .resources.sentence_segmenter import AmharicSentenceSegmenter

Record = Dict[str, Any]


def _require_audio_deps():
    try:
        import librosa  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "This operation needs the optional 'speech' extra: "
            "pip install amharicNLP[speech]"
        ) from exc


@dataclass
class SplitResult:
    train: List[Record] = field(default_factory=list)
    validation: List[Record] = field(default_factory=list)
    test: List[Record] = field(default_factory=list)


class AmharicAudioDatasetProcessor:
    """
        proc = AmharicAudioDatasetProcessor()

        # Metadata-only, no audio libs needed:
        kept, dropped = proc.filter_by_duration(records, min_duration=1.0, max_duration=15.0)
        records = proc.normalize_transcripts(records, text_field="text")
        split = proc.train_val_test_split(records, group_field="speaker_id")

        # Needs pip install amharicNLP[speech]:
        duration_s = proc.get_duration("clip001.wav")
        proc.trim_silence("clip001.wav", "clip001_trimmed.wav")
        proc.resample("clip001.wav", "clip001_16k.wav", target_sr=16000)
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self._normalizer = AmharicNormalizer()
        self._segmenter = AmharicSentenceSegmenter()

    # ---- Audio-file operations (need the 'speech' extra) --------------

    @staticmethod
    def get_duration(audio_path: str) -> float:
        """Duration in seconds, via soundfile (fast -- reads the header,
        not the whole file)."""
        _require_audio_deps()
        import soundfile as sf
        info = sf.info(audio_path)
        return info.frames / float(info.samplerate)

    @staticmethod
    def get_sample_rate(audio_path: str) -> int:
        _require_audio_deps()
        import soundfile as sf
        return sf.info(audio_path).samplerate

    def resample(self, input_path: str, output_path: str, target_sr: Optional[int] = None) -> str:
        """Resample an audio file to `target_sr` (default: this
        processor's target_sample_rate, e.g. 16kHz for most ASR models)
        and write it to `output_path`. Returns output_path."""
        _require_audio_deps()
        import librosa
        import soundfile as sf

        sr = target_sr or self.target_sample_rate
        audio, _ = librosa.load(input_path, sr=sr, mono=True)
        sf.write(output_path, audio, sr)
        return output_path

    @staticmethod
    def trim_silence(input_path: str, output_path: str, top_db: float = 30.0) -> str:
        """Trim leading/trailing silence (librosa.effects.trim) and write
        the result to output_path. `top_db`: the threshold (dB below peak)
        below which audio is considered silence -- lower trims more
        aggressively. Returns output_path."""
        _require_audio_deps()
        import librosa
        import soundfile as sf

        audio, sr = librosa.load(input_path, sr=None, mono=True)
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        sf.write(output_path, trimmed, sr)
        return output_path

    @staticmethod
    def remove_internal_silence(input_path: str, output_path: str, top_db: float = 30.0,
                                  min_silence_s: float = 0.5) -> str:
        """
        Remove silent gaps *inside* the clip (not just leading/trailing),
        e.g. a long pause mid-recording -- splits on silence via
        librosa.effects.split and concatenates the non-silent segments.
        `min_silence_s`: silent gaps shorter than this are left in place
        (avoids chopping up natural micro-pauses between words).
        """
        _require_audio_deps()
        import numpy as np
        import librosa
        import soundfile as sf

        audio, sr = librosa.load(input_path, sr=None, mono=True)
        intervals = librosa.effects.split(audio, top_db=top_db)
        min_silence_samples = int(min_silence_s * sr)

        if len(intervals) == 0:
            sf.write(output_path, audio, sr)
            return output_path

        segments = [audio[start:end] for start, end in intervals]
        merged = [segments[0]]
        for (prev_start, prev_end), (start, end), seg in zip(intervals, intervals[1:], segments[1:]):
            gap = start - prev_end
            if gap < min_silence_samples:
                # Small enough gap: keep the original audio in between
                # instead of hard-cutting, so words don't run together.
                merged.append(audio[prev_end:start])
            merged.append(seg)

        result = np.concatenate(merged) if len(merged) > 1 else merged[0]
        sf.write(output_path, result, sr)
        return output_path

    # ---- Metadata-only operations (no extra deps) ----------------------

    def filter_by_duration(self, records: Sequence[Record],
                             min_duration: Optional[float] = None,
                             max_duration: Optional[float] = None,
                             duration_field: str = "duration_s",
                             compute_missing: bool = False,
                             audio_field: str = "audio_path") -> Tuple[List[Record], List[int]]:
        """
        Filter records by duration. Prefers an already-present
        `duration_field` (cheap, no file I/O); if a record is missing it
        and `compute_missing=True`, computes it from the audio file at
        `audio_field` (needs the 'speech' extra) -- otherwise the record
        is kept as-is (duration unknown != too long/short) with a note
        that it wasn't checked.

        Returns (kept_records, dropped_indices) -- dropped_indices index
        into the original `records` sequence.
        """
        kept: List[Record] = []
        dropped: List[int] = []
        for i, rec in enumerate(records):
            duration = rec.get(duration_field)
            if duration is None and compute_missing and rec.get(audio_field):
                duration = self.get_duration(rec[audio_field])
            if duration is None:
                kept.append(rec)
                continue
            if min_duration is not None and duration < min_duration:
                dropped.append(i)
                continue
            if max_duration is not None and duration > max_duration:
                dropped.append(i)
                continue
            kept.append(rec)
        return kept, dropped

    def normalize_transcripts(self, records: Sequence[Record], text_field: str = "text",
                                new_field: Optional[str] = None, **normalize_kwargs) -> List[Record]:
        """
        Normalize the transcript column via AmharicNormalizer.normalize_text
        -- the same normalization used by the text pipeline, so an
        STT/TTS transcript and an LLM/sentiment text sample go through
        identical Amharic-specific cleanup rather than a separate
        one-off implementation. Writes to `new_field` (default: overwrite
        `text_field` in place) and returns new record dicts (originals
        untouched); all other fields pass through.
        """
        target = new_field or text_field
        result = []
        for rec in records:
            new_rec = dict(rec)
            text = rec.get(text_field)
            if text:
                new_rec[target] = self._normalizer.normalize_text(text, **normalize_kwargs)
            result.append(new_rec)
        return result

    def segment_transcript(self, text: str) -> List[str]:
        """Split a long transcript into sentences (AmharicSentenceSegmenter)
        -- useful when preparing long-form audio for segment-level
        alignment, since forced-alignment tools generally work sentence-
        or utterance-at-a-time rather than on an entire recording."""
        return self._segmenter.split(text)

    def train_val_test_split(self, records: Sequence[Record], train: float = 0.8,
                               validation: float = 0.1, test: float = 0.1,
                               seed: int = 42, group_field: Optional[str] = None) -> SplitResult:
        """
        Split records into train/validation/test.

        `group_field` (e.g. "speaker_id"): when given, every record
        sharing the same value stays entirely within one split -- this
        matters for speech data specifically, since randomly splitting
        individual utterances lets the same speaker's voice appear in
        both train and test, which inflates eval numbers (the model
        partly memorizes the speaker, not the task). Without
        `group_field`, records are split independently.
        """
        ratios_sum = train + validation + test
        if abs(ratios_sum - 1.0) > 1e-6:
            raise ValueError(f"train + validation + test must sum to 1.0, got {ratios_sum}")

        rng = random.Random(seed)

        if group_field is None:
            items = list(records)
            rng.shuffle(items)
            n = len(items)
            n_train = int(round(n * train))
            n_val = int(round(n * validation))
            return SplitResult(
                train=items[:n_train],
                validation=items[n_train:n_train + n_val],
                test=items[n_train + n_val:],
            )

        groups: Dict[Any, List[Record]] = {}
        for rec in records:
            key = rec.get(group_field)
            groups.setdefault(key, []).append(rec)

        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        n_groups = len(group_keys)
        n_train_groups = int(round(n_groups * train))
        n_val_groups = int(round(n_groups * validation))

        result = SplitResult()
        for key in group_keys[:n_train_groups]:
            result.train.extend(groups[key])
        for key in group_keys[n_train_groups:n_train_groups + n_val_groups]:
            result.validation.extend(groups[key])
        for key in group_keys[n_train_groups + n_val_groups:]:
            result.test.extend(groups[key])
        return result

    @staticmethod
    def validate_audio_quality(records: Sequence[Record], audio_field: str = "audio_path",
                                 min_duration: float = 0.1, max_silence_ratio: float = 0.95) -> List[int]:
        """
        Basic audio-quality checks needing the 'speech' extra: flags
        records whose audio file is missing/unreadable, is shorter than
        `min_duration` seconds, or is mostly silence (fraction of samples
        below a low amplitude threshold exceeds `max_silence_ratio` --
        catches near-empty/corrupted recordings, not a full loudness/SNR
        analysis). Returns indices of flagged records.
        """
        _require_audio_deps()
        import os
        import numpy as np
        import librosa

        flagged = []
        for i, rec in enumerate(records):
            path = rec.get(audio_field)
            if not path or not os.path.exists(path):
                flagged.append(i)
                continue
            try:
                audio, sr = librosa.load(path, sr=None, mono=True)
            except Exception:
                flagged.append(i)
                continue
            duration = len(audio) / float(sr) if sr else 0.0
            if duration < min_duration:
                flagged.append(i)
                continue
            if len(audio) == 0:
                flagged.append(i)
                continue
            silence_ratio = float(np.mean(np.abs(audio) < 1e-3))
            if silence_ratio > max_silence_ratio:
                flagged.append(i)
        return flagged
