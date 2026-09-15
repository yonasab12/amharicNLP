"""Tests for the Bring-Your-Own-Data / Bring-Your-Own-Model layer."""

import json

import pytest

from amharicNLP._optional import OptionalDependencyError, have, require
from amharicNLP.inspection import DatasetReport, amharic_ratio, inspect_dataset
from amharicNLP.io_adapters import export, to_records, to_records_dict
from amharicNLP.processor import AmharicDatasetProcessor, ProcessingResult


# ---------------------------------------------------------------- optional deps

class TestOptionalDependencies:
    def test_have_returns_bool(self):
        assert have("json") is True
        assert have("definitely_not_a_real_module_xyz") is False

    def test_require_returns_module_when_present(self):
        assert require("json").dumps({"a": 1}) == '{"a": 1}'

    def test_require_raises_with_install_hint(self):
        with pytest.raises(OptionalDependencyError) as exc:
            require("definitely_not_a_real_module_xyz")
        assert "pip install" in str(exc.value)

    def test_require_error_is_an_importerror(self):
        # Existing `except ImportError` handlers must keep working.
        with pytest.raises(ImportError):
            require("definitely_not_a_real_module_xyz")

    def test_require_mentions_the_feature_that_needed_it(self):
        with pytest.raises(OptionalDependencyError) as exc:
            require("definitely_not_a_real_module_xyz", "Parquet export")
        assert "Parquet export" in str(exc.value)

    def test_known_package_suggests_extra(self):
        if have("torch"):
            pytest.skip("torch installed; can't test its missing-dep message")
        with pytest.raises(OptionalDependencyError) as exc:
            require("torch")
        assert "amharicNLP[neural]" in str(exc.value)


# ------------------------------------------------------------------- BYOD input

class TestToRecords:
    def test_single_string_becomes_one_record(self):
        assert to_records("ሰላም ለዓለም") == [{"text": "ሰላም ለዓለም"}]

    def test_list_of_strings(self):
        assert to_records(["ሰላም", "ደህና"]) == [{"text": "ሰላም"}, {"text": "ደህና"}]

    def test_list_of_dicts_passes_through(self):
        data = [{"text": "ሰላም", "label": "pos"}]
        assert to_records(data) == data

    def test_list_of_dicts_is_copied_not_mutated(self):
        data = [{"text": "ሰላም"}]
        out = to_records(data)
        out[0]["text"] = "changed"
        assert data[0]["text"] == "ሰላም"

    def test_column_oriented_dict(self):
        out = to_records({"text": ["a", "b"], "label": [1, 2]})
        assert out == [{"text": "a", "label": 1}, {"text": "b", "label": 2}]

    def test_column_oriented_dict_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatched"):
            to_records({"text": ["a", "b"], "label": [1]})

    def test_single_record_dict(self):
        assert to_records({"text": "ሰላም", "label": "x"}) == [{"text": "ሰላም", "label": "x"}]

    def test_custom_text_field_name(self):
        assert to_records(["ሰላም"], text_field="sentence") == [{"sentence": "ሰላም"}]

    def test_dataframe(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame([{"text": "ሰላም", "speaker_id": "s1"}])
        assert to_records(df) == [{"text": "ሰላም", "speaker_id": "s1"}]

    def test_reads_jsonl_file_path(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text('{"text": "ሰላም"}\n', encoding="utf-8")
        assert to_records(str(p)) == [{"text": "ሰላም"}]

    def test_reads_csv_file_path(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text("text,label\nሰላም,pos\n", encoding="utf-8")
        assert to_records(str(p)) == [{"text": "ሰላም", "label": "pos"}]

    def test_nonexistent_path_treated_as_text_not_file(self):
        # An Amharic sentence must never be mistaken for a file path.
        assert to_records("ይህ ጽሑፍ ነው") == [{"text": "ይህ ጽሑፍ ነው"}]

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            to_records(42)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            to_records(None)

    def test_non_str_non_dict_item_raises(self):
        with pytest.raises(TypeError, match="Item 1"):
            to_records(["ok", 42])

    def test_to_records_dict_wraps_single_split(self):
        assert to_records_dict(["ሰላም"]) == {"train": [{"text": "ሰላም"}]}

    def test_to_records_dict_handles_split_mapping(self):
        out = to_records_dict({"train": [{"text": "a"}], "test": [{"text": "b"}]})
        assert set(out) == {"train", "test"}


# ------------------------------------------------------------------ export

class TestExport:
    records = [
        {"text": "ሰላም", "label": "pos", "toks": ["ሰ", "ላም"]},
        {"text": "ደህና", "label": "neg"},
    ]

    def test_records_roundtrip(self):
        assert export(self.records, "records") == self.records

    def test_dict_is_column_oriented_and_pads_missing(self):
        out = export(self.records, "dict")
        assert out["text"] == ["ሰላም", "ደህና"]
        assert out["toks"][1] is None   # ragged column padded, not dropped

    def test_jsonl(self, tmp_path):
        p = tmp_path / "o.jsonl"
        export(self.records, "jsonl", path=str(p))
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert json.loads(lines[0])["text"] == "ሰላም"

    def test_json(self, tmp_path):
        p = tmp_path / "o.json"
        export(self.records, "json", path=str(p))
        assert json.loads(p.read_text(encoding="utf-8"))[0]["label"] == "pos"

    def test_csv_includes_columns_missing_from_first_record(self, tmp_path):
        p = tmp_path / "o.csv"
        export([{"a": 1}, {"a": 2, "b": 3}], "csv", path=str(p))
        assert "b" in p.read_text(encoding="utf-8").split("\n")[0]

    def test_txt_writes_text_field_only(self, tmp_path):
        p = tmp_path / "o.txt"
        export(self.records, "txt", path=str(p), text_field="text")
        assert p.read_text(encoding="utf-8").strip().split("\n") == ["ሰላም", "ደህና"]

    def test_file_format_without_path_raises(self):
        with pytest.raises(ValueError, match="writes a file"):
            export(self.records, "jsonl")

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown export format"):
            export(self.records, "xml")

    def test_dataframe(self):
        pytest.importorskip("pandas")
        assert export(self.records, "dataframe").shape[0] == 2


# --------------------------------------------------------------- inspection

class TestInspection:
    def test_amharic_ratio_ignores_digits_and_punctuation(self):
        assert amharic_ratio("አዲስ አበባ 2016") == 1.0

    def test_amharic_ratio_on_latin_text(self):
        assert amharic_ratio("hello") == 0.0

    def test_amharic_ratio_empty(self):
        assert amharic_ratio("") == 0.0

    def test_counts_and_columns(self):
        recs = [{"text": "ሰላም ለዓለም", "src": "a"}, {"text": "ደህና ነህ"}]
        r = inspect_dataset(recs)
        assert r.total_samples == 2
        assert r.valid_samples == 2
        assert set(r.columns) == {"text", "src"}
        assert r.missing_per_column["src"] == 1

    def test_detects_empty_and_duplicates(self):
        recs = [{"text": "ሰላም"}, {"text": "ሰላም"}, {"text": "   "}]
        r = inspect_dataset(recs)
        assert r.empty_text_samples == 1
        assert r.duplicate_samples == 1
        assert r.invalid_samples == 1

    def test_label_distribution_uses_user_labels(self):
        recs = [{"t": "ሀ", "y": "ስፖርት"}, {"t": "ለ", "y": "ፖለቲካ"}, {"t": "ሐ", "y": "ስፖርት"}]
        r = inspect_dataset(recs, text_column="t", label_column="y")
        assert r.label_distribution == {"ስፖርት": 2, "ፖለቲካ": 1}
        assert set(r.unique_labels) == {"ስፖርት", "ፖለቲካ"}

    def test_single_label_warns(self):
        recs = [{"text": "ሰላም ለዓለም", "y": "a"}, {"text": "ደህና ነህ", "y": "a"}]
        r = inspect_dataset(recs, label_column="y")
        assert any("one unique label" in w for w in r.warnings)

    def test_imbalance_ratio(self):
        recs = [{"text": f"ጽሑፍ ቁጥር {i}", "y": "a"} for i in range(20)]
        recs.append({"text": "ብቸኛ", "y": "b"})
        r = inspect_dataset(recs, label_column="y")
        assert r.label_imbalance_ratio == 20.0
        assert any("imbalance" in w.lower() for w in r.warnings)

    def test_language_contamination_warning(self):
        recs = [{"text": "this is english text here"} for _ in range(10)]
        r = inspect_dataset(recs)
        assert r.mostly_non_amharic_samples == 10
        assert any("contamination" in w for w in r.warnings)

    def test_empty_dataset(self):
        r = inspect_dataset([])
        assert r.total_samples == 0
        assert any("empty" in w.lower() for w in r.warnings)

    def test_to_dict_and_str(self):
        r = inspect_dataset([{"text": "ሰላም"}])
        assert isinstance(r.to_dict(), dict)
        assert "Dataset inspection report" in str(r)

    def test_does_not_mutate_input(self):
        recs = [{"text": "ሰላም"}]
        inspect_dataset(recs)
        assert recs == [{"text": "ሰላም"}]


# ---------------------------------------------------------------- processor

class TestProcessorBasics:
    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            AmharicDatasetProcessor(task="not_a_task")

    def test_destructive_ops_off_by_default(self):
        p = AmharicDatasetProcessor()
        assert not p.do_stem and not p.do_remove_stopwords and not p.do_tokenize

    def test_original_column_preserved(self):
        p = AmharicDatasetProcessor()
        r = p.process([{"text": "ሰላም   ለዓለም"}], text_column="text")
        assert r.records[0]["text"] == "ሰላም   ለዓለም"     # untouched
        assert "text_clean" in r.records[0]               # result added separately

    def test_inplace_columns_overwrites(self):
        p = AmharicDatasetProcessor(inplace_columns=True)
        r = p.process([{"text": "ሰላም   ለዓለም"}], text_column="text")
        assert "text_clean" not in r.records[0]
        assert r.records[0]["text"] != "ሰላም   ለዓለም"

    def test_metadata_columns_preserved(self):
        p = AmharicDatasetProcessor()
        r = p.process(
            [{"text": "ሰላም", "speaker_id": "s1", "domain": "news"}],
            text_column="text", metadata_columns=["speaker_id", "domain"],
        )
        assert r.records[0]["speaker_id"] == "s1"
        assert r.records[0]["domain"] == "news"

    def test_missing_text_column_raises_with_available_columns(self):
        p = AmharicDatasetProcessor()
        with pytest.raises(KeyError, match="available columns|Available columns"):
            p.process([{"sentence": "ሰላም"}], text_column="text")

    def test_missing_metadata_column_raises(self):
        p = AmharicDatasetProcessor()
        with pytest.raises(KeyError):
            p.process([{"text": "ሰላም"}], text_column="text", metadata_columns=["nope"])

    def test_missing_label_column_raises(self):
        p = AmharicDatasetProcessor()
        with pytest.raises(KeyError):
            p.process([{"text": "ሰላም"}], text_column="text", label_column="nope")

    def test_empty_input(self):
        r = AmharicDatasetProcessor().process([], text_column="text")
        assert len(r) == 0
        assert any("empty" in w.lower() for w in r.warnings)

    def test_opt_in_tokenize_adds_column(self):
        p = AmharicDatasetProcessor(tokenize=True)
        r = p.process([{"text": "ሰላም ለዓለም"}], text_column="text")
        assert isinstance(r.records[0]["tokens"], list)

    def test_opt_in_segment_adds_sentences(self):
        p = AmharicDatasetProcessor(segment_sentences=True)
        r = p.process([{"text": "ሰላም። ደህና ነህ።"}], text_column="text")
        assert len(r.records[0]["sentences"]) == 2

    def test_drop_invalid_off_by_default_keeps_data(self):
        p = AmharicDatasetProcessor(min_length=5)
        r = p.process([{"text": "ሀ"}], text_column="text")
        assert len(r) == 1 and r.dropped == 0

    def test_drop_invalid_on_removes_and_counts(self):
        p = AmharicDatasetProcessor(min_length=5, drop_invalid=True)
        r = p.process([{"text": "ሀ"}, {"text": "ሰላም ለዓለም ነው"}], text_column="text")
        assert len(r) == 1 and r.dropped == 1


class TestPreserveLatin:
    """Cleaning strips non-Ethiopic text, which is destructive for
    code-switched/translation data -- so it's task-dependent."""

    def test_text_fidelity_tasks_keep_latin(self):
        for task in ("llm", "asr", "tts"):
            p = AmharicDatasetProcessor(task=task)
            assert "Hello" in p.process_text("Hello ሰላም")["text"], task

    def test_other_tasks_strip_latin(self):
        for task in ("classification", "ner", "generic"):
            p = AmharicDatasetProcessor(task=task)
            assert "Hello" not in p.process_text("Hello ሰላም")["text"], task

    def test_explicit_override_wins(self):
        p = AmharicDatasetProcessor(task="classification", preserve_latin=True)
        assert "Hello" in p.process_text("Hello ሰላም")["text"]


class TestCustomLabels:
    def test_labels_discovered_from_data_not_hardcoded(self):
        data = [{"t": "ሀ", "y": "ጥያቄ"}, {"t": "ለ", "y": "ቅሬታ"}, {"t": "ሐ", "y": "ጥያቄ"}]
        r = AmharicDatasetProcessor(task="classification").process(
            data, text_column="t", label_column="y"
        )
        assert r.labels == ["ጥያቄ", "ቅሬታ"]
        assert r.label2id == {"ጥያቄ": 0, "ቅሬታ": 1}
        assert r.id2label == {0: "ጥያቄ", 1: "ቅሬታ"}

    def test_no_sentiment_vocabulary_is_assumed(self):
        data = [{"t": "ሀ", "y": "politics"}, {"t": "ለ", "y": "sports"}]
        r = AmharicDatasetProcessor(task="classification").process(
            data, text_column="t", label_column="y"
        )
        assert set(r.labels) == {"politics", "sports"}
        assert "positive" not in r.labels and "neutral" not in r.labels

    def test_classification_without_labels_warns(self):
        r = AmharicDatasetProcessor(task="classification").process(
            [{"text": "ሰላም"}], text_column="text"
        )
        assert any("no label_column" in w for w in r.warnings)


class TestNER:
    def test_tokens_and_tags_stay_aligned(self):
        data = [{"tokens": ["አበበ", "ወደ", "አዲስ", "አበባ"],
                  "ner_tags": ["B-PER", "O", "B-LOC", "I-LOC"]}]
        r = AmharicDatasetProcessor(task="ner").process(
            data, tokens_column="tokens", tags_column="ner_tags"
        )
        rec = r.records[0]
        assert len(rec["tokens"]) == len(rec["ner_tags"]) == 4

    def test_custom_entity_types_discovered(self):
        data = [{"tokens": ["ሀ", "ለ"], "ner_tags": ["B-CUSTOM_ENTITY", "O"]}]
        r = AmharicDatasetProcessor(task="ner").process(
            data, tokens_column="tokens", tags_column="ner_tags"
        )
        assert "B-CUSTOM_ENTITY" in r.labels

    def test_misaligned_annotation_dropped_not_corrupted(self):
        data = [{"tokens": ["ሀ", "ለ"], "ner_tags": ["O"]}]
        r = AmharicDatasetProcessor(task="ner").process(
            data, tokens_column="tokens", tags_column="ner_tags"
        )
        assert len(r) == 0 and r.dropped == 1

    def test_token_normalization_never_empties_a_token(self):
        # An emptied token would shift every subsequent tag.
        data = [{"tokens": ["@@@", "ሰላም"], "ner_tags": ["O", "B-X"]}]
        r = AmharicDatasetProcessor(task="ner").process(
            data, tokens_column="tokens", tags_column="ner_tags"
        )
        assert all(t for t in r.records[0]["tokens"])

    def test_metadata_preserved_on_ner_records(self):
        data = [{"tokens": ["ሀ"], "ner_tags": ["O"], "doc_id": 7}]
        r = AmharicDatasetProcessor(task="ner").process(
            data, tokens_column="tokens", tags_column="ner_tags"
        )
        assert r.records[0]["doc_id"] == 7

    def test_missing_tokens_column_raises(self):
        with pytest.raises(KeyError):
            AmharicDatasetProcessor(task="ner").process(
                [{"text": "ሰላም"}], tokens_column="tokens", tags_column="tags"
            )


class TestTaskWarnings:
    def test_stemming_on_asr_warns(self):
        p = AmharicDatasetProcessor(task="asr", stem=True)
        assert any("degrade ASR" in w for w in p._config_warnings)

    def test_stemming_on_classification_does_not_warn(self):
        p = AmharicDatasetProcessor(task="classification", stem=True)
        assert p._config_warnings == []

    def test_asr_without_audio_column_warns(self):
        r = AmharicDatasetProcessor(task="asr").process(
            [{"transcription": "ሰላም"}], text_column="transcription"
        )
        assert any("no audio column" in w for w in r.warnings)

    def test_llm_instruction_data_points_at_validator(self):
        r = AmharicDatasetProcessor(task="llm").process(
            [{"instruction": "ተርጉም", "output": "ሰላም", "text": "x"}], text_column="text"
        )
        assert any("AmharicLLMDatasetValidator" in w for w in r.warnings)


class TestProcessingResult:
    def _result(self, n=40):
        data = [{"text": f"ጽሑፍ ቁጥር {i} ነው", "label": "pos" if i % 4 else "neg"}
                for i in range(n)]
        return AmharicDatasetProcessor(task="classification").process(
            data, text_column="text", label_column="label", inspect=False
        )

    def test_len_iter_getitem(self):
        r = self._result(5)
        assert len(r) == 5
        assert r[0] is r.records[0]
        assert len(list(r)) == 5

    def test_split_three_way(self):
        s = self._result().split((0.8, 0.1, 0.1))
        assert set(s) == {"train", "validation", "test"}
        assert sum(len(v) for v in s.values()) == 40

    def test_split_two_way(self):
        s = self._result().split((0.8, 0.2))
        assert set(s) == {"train", "test"}

    def test_split_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            self._result().split((0.5, 0.2, 0.2))

    def test_stratified_split_preserves_label_balance(self):
        s = self._result().split((0.5, 0.5), stratify=True)
        for part in s.values():
            labels = [rec["label"] for rec in part]
            assert "neg" in labels and "pos" in labels

    def test_stratify_without_label_column_raises(self):
        r = AmharicDatasetProcessor().process([{"text": "ሰላም"}], text_column="text")
        with pytest.raises(ValueError, match="label_column"):
            r.split((0.5, 0.5), stratify=True)

    def test_split_is_deterministic_for_a_seed(self):
        a = self._result().split(seed=7)["train"]
        b = self._result().split(seed=7)["train"]
        assert [r["text"] for r in a] == [r["text"] for r in b]

    def test_export_infers_format_from_extension(self, tmp_path):
        p = tmp_path / "out.jsonl"
        self._result(3).export(str(p))
        assert p.exists() and p.read_text(encoding="utf-8").count("\n") == 3

    def test_to_dataframe(self):
        pytest.importorskip("pandas")
        assert self._result(5).to_dataframe().shape[0] == 5

    def test_summary_is_a_string(self):
        assert "ProcessingResult" in self._result(3).summary()


class TestInspectAndValidateEntryPoints:
    def test_inspect_does_not_process(self):
        p = AmharicDatasetProcessor()
        report = p.inspect([{"text": "ሰላም   ለዓለም"}], text_column="text")
        assert isinstance(report, DatasetReport)
        assert report.total_samples == 1

    def test_validate_adds_task_warnings(self):
        p = AmharicDatasetProcessor(task="classification")
        report = p.validate([{"text": "ሰላም"}], text_column="text")
        assert any("no label_column" in w for w in report.warnings)
