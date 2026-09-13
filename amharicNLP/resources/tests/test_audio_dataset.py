import os

import pytest

from amharicNLP.audio_dataset import AmharicAudioDatasetProcessor, SplitResult

librosa = pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")
np = pytest.importorskip("numpy")


@pytest.fixture
def proc():
    return AmharicAudioDatasetProcessor(target_sample_rate=16000)


@pytest.fixture
def silence_tone_silence_wav(tmp_path):
    """1s silence + 1s 440Hz tone + 1s silence at 22050 Hz."""
    sr = 22050
    silence = np.zeros(sr, dtype=np.float32)
    t = np.linspace(0, 1, sr, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio = np.concatenate([silence, tone, silence])
    path = tmp_path / "clip.wav"
    sf.write(str(path), audio, sr)
    return str(path), sr


@pytest.fixture
def silent_wav(tmp_path):
    sr = 16000
    path = tmp_path / "silent.wav"
    sf.write(str(path), np.zeros(sr, dtype=np.float32), sr)
    return str(path)


class TestAudioFileOperations:
    def test_get_duration(self, proc, silence_tone_silence_wav):
        path, sr = silence_tone_silence_wav
        assert proc.get_duration(path) == pytest.approx(3.0, abs=0.01)

    def test_get_sample_rate(self, proc, silence_tone_silence_wav):
        path, sr = silence_tone_silence_wav
        assert proc.get_sample_rate(path) == sr

    def test_trim_silence_shortens_clip(self, proc, silence_tone_silence_wav, tmp_path):
        path, _ = silence_tone_silence_wav
        out = str(tmp_path / "trimmed.wav")
        proc.trim_silence(path, out, top_db=20)
        trimmed_duration = proc.get_duration(out)
        assert trimmed_duration < 2.0  # leading/trailing silence removed
        assert trimmed_duration > 0.5  # tone itself preserved

    def test_resample_changes_sample_rate(self, proc, silence_tone_silence_wav, tmp_path):
        path, _ = silence_tone_silence_wav
        out = str(tmp_path / "resampled.wav")
        proc.resample(path, out, target_sr=16000)
        assert proc.get_sample_rate(out) == 16000

    def test_resample_uses_processor_default(self, silence_tone_silence_wav, tmp_path):
        proc = AmharicAudioDatasetProcessor(target_sample_rate=8000)
        path, _ = silence_tone_silence_wav
        out = str(tmp_path / "resampled.wav")
        proc.resample(path, out)
        assert proc.get_sample_rate(out) == 8000

    def test_remove_internal_silence_shortens_clip(self, proc, tmp_path):
        sr = 22050
        t = np.linspace(0, 1, sr, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio = np.concatenate([tone, np.zeros(sr, dtype=np.float32), tone])
        path = str(tmp_path / "gap.wav")
        sf.write(path, audio, sr)

        out = str(tmp_path / "nogap.wav")
        proc.remove_internal_silence(path, out, top_db=20, min_silence_s=0.3)
        assert proc.get_duration(out) < proc.get_duration(path)

    def test_missing_speech_deps_raises_clear_error(self, monkeypatch, silence_tone_silence_wav):
        import amharicNLP.audio_dataset as mod
        path, _ = silence_tone_silence_wav

        def _boom():
            raise ImportError(
                "This operation needs the optional 'speech' extra: "
                "pip install amharicNLP[speech]"
            )
        monkeypatch.setattr(mod, "_require_audio_deps", _boom)
        with pytest.raises(ImportError, match="speech"):
            AmharicAudioDatasetProcessor.get_duration(path)


class TestValidateAudioQuality:
    def test_flags_missing_file(self, proc):
        assert proc.validate_audio_quality([{"audio_path": "does_not_exist.wav"}]) == [0]

    def test_flags_silent_audio(self, proc, silent_wav):
        assert proc.validate_audio_quality([{"audio_path": silent_wav}]) == [0]

    def test_does_not_flag_normal_audio(self, proc, silence_tone_silence_wav):
        path, _ = silence_tone_silence_wav
        assert proc.validate_audio_quality([{"audio_path": path}], max_silence_ratio=0.9) == []

    def test_flags_too_short_audio(self, proc, tmp_path):
        sr = 16000
        path = str(tmp_path / "tiny.wav")
        sf.write(path, np.random.uniform(-0.5, 0.5, int(sr * 0.05)).astype(np.float32), sr)
        assert proc.validate_audio_quality([{"audio_path": path}], min_duration=0.1) == [0]


class TestMetadataOnlyOperations:
    def test_filter_by_duration_keeps_within_range(self, proc):
        records = [
            {"text": "a", "duration_s": 2.0},
            {"text": "b", "duration_s": 0.05},
            {"text": "c", "duration_s": 30.0},
            {"text": "d", "duration_s": 3.0},
        ]
        kept, dropped = proc.filter_by_duration(records, min_duration=0.5, max_duration=15.0)
        assert [r["text"] for r in kept] == ["a", "d"]
        assert dropped == [1, 2]

    def test_filter_by_duration_keeps_records_missing_duration_by_default(self, proc):
        records = [{"text": "a"}]
        kept, dropped = proc.filter_by_duration(records, min_duration=0.5)
        assert kept == records
        assert dropped == []

    def test_filter_by_duration_computes_missing_when_asked(self, proc, silence_tone_silence_wav):
        path, _ = silence_tone_silence_wav
        records = [{"audio_path": path}]
        kept, dropped = proc.filter_by_duration(
            records, min_duration=10.0, compute_missing=True,
        )
        assert dropped == [0]

    def test_normalize_transcripts_applies_normalizer(self, proc):
        records = [{"text": "ደሓናናናና ነኝ", "id": 1}]
        result = proc.normalize_transcripts(records, collapse_repeated_chars=True)
        assert "ናናናና" not in result[0]["text"]
        assert result[0]["id"] == 1  # other fields untouched

    def test_normalize_transcripts_does_not_mutate_input(self, proc):
        records = [{"text": "ደሓናናናና"}]
        proc.normalize_transcripts(records, collapse_repeated_chars=True)
        assert records[0]["text"] == "ደሓናናናና"

    def test_normalize_transcripts_writes_to_new_field(self, proc):
        records = [{"text": "ደሓናናናና"}]
        result = proc.normalize_transcripts(
            records, new_field="text_clean", collapse_repeated_chars=True,
        )
        assert "text" in result[0]
        assert "text_clean" in result[0]

    def test_segment_transcript(self, proc):
        assert proc.segment_transcript("ሰላም ለዓለም። እንዴት ነህ?") == ["ሰላም ለዓለም።", "እንዴት ነህ?"]


class TestTrainValTestSplit:
    def test_ratios_must_sum_to_one(self, proc):
        with pytest.raises(ValueError):
            proc.train_val_test_split([{"text": "a"}], train=0.5, validation=0.5, test=0.5)

    def test_split_without_grouping_covers_all_records(self, proc):
        records = [{"id": i} for i in range(20)]
        split = proc.train_val_test_split(records, seed=1)
        assert isinstance(split, SplitResult)
        total = len(split.train) + len(split.validation) + len(split.test)
        assert total == 20

    def test_split_is_deterministic_given_seed(self, proc):
        records = [{"id": i} for i in range(20)]
        split_a = proc.train_val_test_split(records, seed=7)
        split_b = proc.train_val_test_split(records, seed=7)
        assert [r["id"] for r in split_a.train] == [r["id"] for r in split_b.train]

    def test_group_field_keeps_speaker_in_one_split(self, proc):
        records = []
        for spk in ["s1", "s2", "s3", "s4", "s5"]:
            for j in range(4):
                records.append({"speaker_id": spk, "clip": j})
        split = proc.train_val_test_split(records, group_field="speaker_id", seed=3)

        train_speakers = {r["speaker_id"] for r in split.train}
        val_speakers = {r["speaker_id"] for r in split.validation}
        test_speakers = {r["speaker_id"] for r in split.test}
        assert not (train_speakers & val_speakers)
        assert not (train_speakers & test_speakers)
        assert not (val_speakers & test_speakers)

    def test_group_split_covers_all_records(self, proc):
        records = []
        for spk in ["s1", "s2", "s3"]:
            for j in range(3):
                records.append({"speaker_id": spk, "clip": j})
        split = proc.train_val_test_split(records, group_field="speaker_id", seed=2)
        total = len(split.train) + len(split.validation) + len(split.test)
        assert total == len(records)
