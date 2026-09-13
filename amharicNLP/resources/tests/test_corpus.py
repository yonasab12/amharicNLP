import json
import os

import pytest

from amharicNLP.corpus import AmharicCorpusProcessor, ValidationReport


@pytest.fixture
def proc():
    return AmharicCorpusProcessor()


class TestLoaders:
    def test_load_txt_skips_blank_lines_by_default(self, tmp_path, proc):
        p = tmp_path / "a.txt"
        p.write_text("ሰላም\n\nደህና ነህ\n", encoding="utf-8")
        assert proc.load_txt(str(p)) == [{"text": "ሰላም"}, {"text": "ደህና ነህ"}]

    def test_load_txt_keeps_blank_lines_when_asked(self, tmp_path, proc):
        p = tmp_path / "a.txt"
        p.write_text("ሰላም\n\n", encoding="utf-8")
        assert proc.load_txt(str(p), skip_blank=False) == [{"text": "ሰላም"}, {"text": ""}]

    def test_load_csv(self, tmp_path, proc):
        p = tmp_path / "a.csv"
        p.write_text("text,label\nሰላም,pos\nደህና,neg\n", encoding="utf-8")
        assert proc.load_csv(str(p)) == [
            {"text": "ሰላም", "label": "pos"},
            {"text": "ደህና", "label": "neg"},
        ]

    def test_load_json_top_level_array(self, tmp_path, proc):
        p = tmp_path / "a.json"
        p.write_text(json.dumps([{"text": "x"}, {"text": "y"}]), encoding="utf-8")
        assert proc.load_json(str(p)) == [{"text": "x"}, {"text": "y"}]

    def test_load_json_wrapped_under_data_key(self, tmp_path, proc):
        p = tmp_path / "a.json"
        p.write_text(json.dumps({"data": [{"text": "x"}]}), encoding="utf-8")
        assert proc.load_json(str(p)) == [{"text": "x"}]

    def test_load_json_explicit_records_key(self, tmp_path, proc):
        p = tmp_path / "a.json"
        p.write_text(json.dumps({"items": [{"text": "x"}]}), encoding="utf-8")
        assert proc.load_json(str(p), records_key="items") == [{"text": "x"}]

    def test_load_json_no_recognizable_key_raises(self, tmp_path, proc):
        p = tmp_path / "a.json"
        p.write_text(json.dumps({"nope": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            proc.load_json(str(p))

    def test_load_jsonl(self, tmp_path, proc):
        p = tmp_path / "a.jsonl"
        p.write_text('{"text": "x"}\n{"text": "y"}\n', encoding="utf-8")
        assert proc.load_jsonl(str(p)) == [{"text": "x"}, {"text": "y"}]

    def test_load_jsonl_bad_line_raises_with_line_number(self, tmp_path, proc):
        p = tmp_path / "a.jsonl"
        p.write_text('{"text": "x"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="line 2"):
            proc.load_jsonl(str(p))

    def test_load_dataframe(self, proc):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame([{"text": "x"}, {"text": "y"}])
        assert proc.load_dataframe(df) == [{"text": "x"}, {"text": "y"}]


class TestFindEmpty:
    def test_finds_missing_none_and_whitespace(self, proc):
        records = [{"text": "ok"}, {"text": ""}, {"text": "   "}, {}, {"text": None}]
        assert proc.find_empty(records) == [1, 2, 3, 4]


class TestFindMissingFields:
    def test_reports_per_field_indices(self, proc):
        records = [
            {"text": "a", "label": "pos"},
            {"text": "b"},
            {"label": "neg"},
        ]
        missing = proc.find_missing_fields(records, ["text", "label"])
        assert missing["text"] == [2]
        assert missing["label"] == [1]


class TestDuplicates:
    def test_exact_duplicates_keep_first(self, proc):
        records = [{"text": "ሰላም"}, {"text": "ሰላም"}, {"text": "ደህና"}, {"text": "ሰላም"}]
        assert proc.find_duplicates(records) == [1, 3]

    def test_exact_duplicates_keep_none(self, proc):
        records = [{"text": "ሰላም"}, {"text": "ሰላም"}, {"text": "ደህና"}]
        assert proc.find_duplicates(records, keep="none") == [0, 1]

    def test_duplicates_ignore_whitespace_and_unicode_form(self, proc):
        import unicodedata
        nfd = unicodedata.normalize("NFD", "ሰላም")
        records = [{"text": "ሰላም"}, {"text": nfd}]
        assert proc.find_duplicates(records) == [1]

    def test_no_duplicates_among_empty_texts(self, proc):
        records = [{"text": ""}, {"text": ""}]
        assert proc.find_duplicates(records) == []

    def test_invalid_keep_raises(self, proc):
        with pytest.raises(ValueError):
            proc.find_duplicates([{"text": "a"}], keep="bogus")


class TestNearDuplicates:
    def test_flags_similar_text(self, proc):
        records = [
            {"text": "ሰላም ለዓለም እንዴት ነህ"},
            {"text": "ሰላም ለዓለም እንዴት ነህ!"},
            {"text": "ፍጹም የተለየ አረፍተ ነገር ነው"},
        ]
        near = proc.find_near_duplicates(records, threshold=0.85)
        assert near == [1]

    def test_respects_exclude_indices(self, proc):
        records = [{"text": "ሰላም ለዓለም"}, {"text": "ሰላም ለዓለም!"}]
        assert proc.find_near_duplicates(records, threshold=0.5, exclude_indices={1}) == []

    def test_invalid_threshold_raises(self, proc):
        with pytest.raises(ValueError):
            proc.find_near_duplicates([{"text": "a"}], threshold=0)
        with pytest.raises(ValueError):
            proc.find_near_duplicates([{"text": "a"}], threshold=1.5)


class TestCorruptText:
    def test_replacement_character_is_corrupt(self, proc):
        assert proc.is_corrupt_text("hello \ufffd world") is True

    def test_control_characters_are_corrupt(self, proc):
        assert proc.is_corrupt_text("hello\x00world") is True

    def test_normal_text_is_not_corrupt(self, proc):
        assert proc.is_corrupt_text("ሰላም ለዓለም") is False

    def test_tabs_and_newlines_are_not_flagged(self, proc):
        assert proc.is_corrupt_text("ሰላም\tለዓለም\n") is False

    def test_empty_text_is_not_corrupt(self, proc):
        assert proc.is_corrupt_text("") is False

    def test_find_corrupt_over_records(self, proc):
        records = [{"text": "ok"}, {"text": "bad \ufffd"}]
        assert proc.find_corrupt(records) == [1]


class TestValidateAndProcess:
    def test_validate_returns_report_without_mutating_input(self, proc):
        records = [{"text": "ሰላም"}, {"text": "ሰላም"}, {"text": ""}]
        report = proc.validate(records)
        assert isinstance(report, ValidationReport)
        assert report.total == 3
        assert len(records) == 3  # unchanged

    def test_process_removes_flagged_records_and_keeps_others_intact(self, proc):
        records = [
            {"text": "ሰላም", "id": 1},
            {"text": "ሰላም", "id": 2},   # exact dup
            {"text": "  ", "id": 3},      # empty
            {"text": "ደህና ነህ", "id": 4},
        ]
        clean, report = proc.process(records)
        assert clean == [{"text": "ሰላም", "id": 1}, {"text": "ደህና ነህ", "id": 4}]
        assert report.kept == 2
        assert report.total == 4

    def test_process_respects_remove_flags(self, proc):
        records = [{"text": "ሰላም"}, {"text": "ሰላም"}]
        clean, report = proc.process(records, remove_duplicates=False)
        assert len(clean) == 2

    def test_process_required_fields_always_dropped(self, proc):
        records = [{"text": "ሰላም", "label": "pos"}, {"text": "ደህና"}]
        clean, report = proc.process(
            records, required_fields=["label"], remove_empty=False,
            remove_duplicates=False, remove_corrupt=False,
        )
        assert clean == [{"text": "ሰላም", "label": "pos"}]

    def test_summary_is_a_string(self, proc):
        report = proc.validate([{"text": "ok"}])
        assert isinstance(report.summary(), str)
        assert "kept: 1/1" in report.summary()
