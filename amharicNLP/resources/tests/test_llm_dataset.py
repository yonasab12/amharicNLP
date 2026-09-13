import pytest

from amharicNLP.llm_dataset import AmharicLLMDatasetValidator, LLMValidationReport


@pytest.fixture
def v():
    return AmharicLLMDatasetValidator()


class TestMissingFields:
    def test_instruction_shape_ok(self, v):
        records = [{"instruction": "ተርጉም", "output": "ሰላም"}]
        assert v.find_missing_fields(records) == []

    def test_empty_instruction_flagged(self, v):
        records = [{"instruction": "", "output": "ሰላም"}]
        assert v.find_missing_fields(records) == [0]

    def test_prompt_completion_shape_ok(self, v):
        records = [{"prompt": "hi", "completion": "hello"}]
        assert v.find_missing_fields(records) == []

    def test_chat_shape_not_flagged_here(self, v):
        records = [{"messages": [{"role": "user", "content": "hi"}]}]
        assert v.find_missing_fields(records) == []

    def test_unrecognized_shape_flagged(self, v):
        records = [{"foo": "bar"}]
        assert v.find_missing_fields(records) == [0]


class TestInvalidConversations:
    def test_valid_chat_not_flagged(self, v):
        records = [{"messages": [
            {"role": "user", "content": "ሰላም"},
            {"role": "assistant", "content": "ሰላም! እንዴት ልረዳህ?"},
        ]}]
        assert v.find_invalid_conversations(records) == []

    def test_empty_turn_list_flagged(self, v):
        assert v.find_invalid_conversations([{"messages": []}]) == [0]

    def test_no_assistant_turn_flagged(self, v):
        records = [{"messages": [{"role": "user", "content": "ሰላም"}]}]
        assert v.find_invalid_conversations(records) == [0]

    def test_empty_content_flagged(self, v):
        records = [{"messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "ok"},
        ]}]
        assert v.find_invalid_conversations(records) == [0]

    def test_unknown_role_flagged(self, v):
        records = [{"messages": [{"role": "narrator", "content": "hi"}]}]
        assert v.find_invalid_conversations(records) == [0]

    def test_sharegpt_shape_supported(self, v):
        records = [{"conversations": [
            {"from": "human", "value": "ሰላም"},
            {"from": "gpt", "value": "ሰላም!"},
        ]}]
        assert v.find_invalid_conversations(records) == []

    def test_non_chat_records_ignored(self, v):
        records = [{"instruction": "x", "output": "y"}]
        assert v.find_invalid_conversations(records) == []


class TestEmptyResponses:
    def test_empty_output_flagged(self, v):
        records = [{"instruction": "x", "output": ""}]
        assert v.find_empty_responses(records) == [0]

    def test_nonempty_output_not_flagged(self, v):
        records = [{"instruction": "x", "output": "ok"}]
        assert v.find_empty_responses(records) == []

    def test_empty_completion_flagged(self, v):
        records = [{"prompt": "x", "completion": "  "}]
        assert v.find_empty_responses(records) == [0]

    def test_chat_shape_uses_last_assistant_turn(self, v):
        records = [{"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ]}]
        assert v.find_empty_responses(records) == [0]


class TestTooLong:
    def test_flags_over_limit(self, v):
        records = [{"instruction": "x", "output": "y" * 100}]
        assert v.find_too_long(records, max_chars=50) == [0]

    def test_does_not_flag_under_limit(self, v):
        records = [{"instruction": "x", "output": "short"}]
        assert v.find_too_long(records, max_chars=50) == []

    def test_uses_constructor_default(self):
        v = AmharicLLMDatasetValidator(max_chars=10)
        records = [{"instruction": "x", "output": "y" * 20}]
        assert v.find_too_long(records) == [0]


class TestDuplicates:
    def test_flags_exact_duplicate_examples(self, v):
        records = [
            {"instruction": "ተርጉም", "output": "ሰላም"},
            {"instruction": "ተርጉም", "output": "ሰላም"},
        ]
        assert v.find_duplicates(records) == [1]

    def test_different_examples_not_flagged(self, v):
        records = [
            {"instruction": "ተርጉም", "output": "ሰላም"},
            {"instruction": "ተርጉም ይህን", "output": "ደህና"},
        ]
        assert v.find_duplicates(records) == []


class TestBrokenUnicode:
    def test_flags_replacement_char(self, v):
        records = [{"instruction": "x", "output": "bad \ufffd text"}]
        assert v.find_broken_unicode(records) == [0]

    def test_clean_text_not_flagged(self, v):
        records = [{"instruction": "x", "output": "ሰላም"}]
        assert v.find_broken_unicode(records) == []


class TestLanguageContamination:
    def test_flags_non_amharic_response_when_long_enough(self, v):
        records = [{
            "instruction": "ስም ንገረኝ",
            "output": "My name is John and I live in the United States doing software work",
        }]
        assert v.find_language_contamination(records) == [0]

    def test_amharic_response_not_flagged(self, v):
        records = [{"instruction": "ስም ንገረኝ", "output": "ስሜ አበበ ይባላል፣ አዲስ አበባ እኖራለሁ።"}]
        assert v.find_language_contamination(records) == []

    def test_short_response_not_flagged_even_if_non_amharic(self, v):
        records = [{"instruction": "x", "output": "yes"}]
        assert v.find_language_contamination(records) == []


class TestPoorQuality:
    def test_flags_response_echoing_instruction(self, v):
        records = [{"instruction": "ይህን ድገም", "output": "ይህን ድገም"}]
        assert v.find_poor_quality(records) == [0]

    def test_flags_low_character_diversity(self, v):
        records = [{"instruction": "x", "output": "aaaaaaaaaaaaaaaaaaaa"}]
        assert v.find_poor_quality(records) == [0]

    def test_normal_response_not_flagged(self, v):
        records = [{"instruction": "ስም ንገረኝ", "output": "ስሜ አበበ ይባላል።"}]
        assert v.find_poor_quality(records) == []


class TestLeakage:
    def test_flags_eval_examples_present_in_train(self, v):
        train = [{"instruction": "ተርጉም", "output": "ሰላም"}]
        eval_ = [
            {"instruction": "ተርጉም", "output": "ሰላም"},
            {"instruction": "X", "output": "Y"},
        ]
        assert v.check_leakage(train, eval_) == [0]

    def test_no_leakage_when_disjoint(self, v):
        train = [{"instruction": "ተርጉም", "output": "ሰላም"}]
        eval_ = [{"instruction": "X", "output": "Y"}]
        assert v.check_leakage(train, eval_) == []


class TestValidateAndProcess:
    def test_validate_returns_report(self, v):
        report = v.validate([{"instruction": "x", "output": "ok"}])
        assert isinstance(report, LLMValidationReport)
        assert report.total == 1
        assert report.kept == 1

    def test_process_removes_flagged_and_preserves_good_examples(self, v):
        examples = [
            {"instruction": "ተርጉም", "output": "ሰላም"},
            {"instruction": "ተርጉም", "output": "ሰላም"},  # dup
            {"instruction": "x", "output": ""},  # empty response
        ]
        clean, report = v.process(examples)
        assert clean == [{"instruction": "ተርጉም", "output": "ሰላም"}]
        assert report.kept == 1

    def test_language_contamination_off_by_default(self, v):
        examples = [{
            "instruction": "ስም ንገረኝ",
            "output": "My name is John and I live in the United States doing software work",
        }]
        clean, report = v.process(examples)
        assert report.language_contamination_indices == []
        assert clean == examples

    def test_poor_quality_on_by_default(self, v):
        examples = [{"instruction": "x", "output": "aaaaaaaaaaaaaaaaaaaa"}]
        clean, report = v.process(examples)
        assert clean == []

    def test_summary_is_string(self, v):
        report = v.validate([{"instruction": "x", "output": "ok"}])
        assert "kept: 1/1" in report.summary()
