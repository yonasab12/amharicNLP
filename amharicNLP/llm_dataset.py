"""
Validation for Amharic LLM training datasets: instruction-tuning,
prompt/completion, and multi-turn conversation formats.

This sits one layer above `AmharicCorpusProcessor` (corpus.py): the
corpus processor validates arbitrary flat records (any text field, any
metadata); this module understands the *shape* an LLM training example
is supposed to have -- an instruction/response pair or a list of chat
turns -- and checks things that only make sense at that level (does this
conversation alternate roles sensibly, is the completion just the prompt
repeated back, does this split leak into that one).

Supported example shapes (auto-detected per record, or set explicitly):

  instruction format:  {"instruction": ..., "input": ..., "output": ...}
  prompt/completion:   {"prompt": ..., "completion": ...}
  chat/messages:       {"messages": [{"role": "user", "content": ...},
                                      {"role": "assistant", "content": ...}, ...]}
  ShareGPT-style:       {"conversations": [{"from": "human", "value": ...},
                                            {"from": "gpt", "value": ...}, ...]}

The package deliberately doesn't force one of these on you (see the
project's core philosophy: bring your own data) -- `AmharicLLMDatasetValidator`
reads whichever shape is present per record.
"""

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .corpus import AmharicCorpusProcessor
from .resources.utils import AmharicLanguageDetector

Record = Dict[str, Any]

# Role names treated as "the model's turn" / "the human's turn" across the
# two conversation shapes this module understands.
_ASSISTANT_ROLES = {"assistant", "gpt", "model", "bot"}
_USER_ROLES = {"user", "human"}
_SYSTEM_ROLES = {"system"}


@dataclass
class LLMValidationReport:
    total: int = 0
    missing_field_indices: List[int] = field(default_factory=list)
    invalid_conversation_indices: List[int] = field(default_factory=list)
    empty_response_indices: List[int] = field(default_factory=list)
    too_long_indices: List[int] = field(default_factory=list)
    duplicate_indices: List[int] = field(default_factory=list)
    broken_unicode_indices: List[int] = field(default_factory=list)
    language_contamination_indices: List[int] = field(default_factory=list)
    poor_quality_indices: List[int] = field(default_factory=list)
    leakage_indices: List[int] = field(default_factory=list)  # only set by check_leakage
    kept: int = 0

    def bad_indices(self) -> Set[int]:
        bad: Set[int] = set()
        for idxs in (
            self.missing_field_indices, self.invalid_conversation_indices,
            self.empty_response_indices, self.too_long_indices,
            self.duplicate_indices, self.broken_unicode_indices,
            self.language_contamination_indices, self.poor_quality_indices,
            self.leakage_indices,
        ):
            bad |= set(idxs)
        return bad

    def summary(self) -> str:
        lines = [f"total examples: {self.total}"]
        for label, idxs in (
            ("missing required fields", self.missing_field_indices),
            ("invalid conversations", self.invalid_conversation_indices),
            ("empty responses", self.empty_response_indices),
            ("extremely long examples", self.too_long_indices),
            ("duplicate examples", self.duplicate_indices),
            ("broken unicode", self.broken_unicode_indices),
            ("language contamination", self.language_contamination_indices),
            ("poor-quality samples", self.poor_quality_indices),
            ("train/eval leakage", self.leakage_indices),
        ):
            if idxs:
                lines.append(f"{label}: {len(idxs)}")
        lines.append(f"kept: {self.kept}/{self.total}")
        return "\n".join(lines)


class AmharicLLMDatasetValidator:
    """
        validator = AmharicLLMDatasetValidator()
        clean, report = validator.process(examples)
        print(report.summary())

    `examples` is a `list[dict]`, one of the shapes documented at module
    level. Records aren't required to all share the same shape -- each is
    inspected independently.
    """

    def __init__(self, max_chars: int = 8000, min_response_chars: int = 1):
        self.max_chars = max_chars
        self.min_response_chars = min_response_chars
        self._corpus = AmharicCorpusProcessor()

    # ---- Shape extraction -------------------------------------------

    @staticmethod
    def _turns_from_messages(record: Record) -> Optional[List[Tuple[str, str]]]:
        """Returns [(role, content), ...] for 'messages' (OpenAI-chat-style)
        or 'conversations' (ShareGPT-style) records, or None if neither
        key is present."""
        if "messages" in record and isinstance(record["messages"], list):
            turns = []
            for m in record["messages"]:
                role = str(m.get("role", "")).lower()
                content = m.get("content", "")
                turns.append((role, content))
            return turns
        if "conversations" in record and isinstance(record["conversations"], list):
            turns = []
            for m in record["conversations"]:
                role = str(m.get("from", "")).lower()
                content = m.get("value", "")
                turns.append((role, content))
            return turns
        return None

    @staticmethod
    def _flat_text(record: Record) -> str:
        """The full example collapsed to one string, for length/dedup/
        language checks -- concatenation of every prompt+response part,
        regardless of shape."""
        turns = AmharicLLMDatasetValidator._turns_from_messages(record)
        if turns is not None:
            return "\n".join(str(c) for _, c in turns)
        parts = [
            str(record.get("instruction", "") or ""),
            str(record.get("input", "") or ""),
            str(record.get("output", "") or ""),
            str(record.get("prompt", "") or ""),
            str(record.get("completion", "") or ""),
        ]
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _response_text(record: Record) -> Optional[str]:
        """The model's expected output, whichever shape the record is in.
        None if the shape isn't recognized at all (handled separately by
        missing-field detection)."""
        turns = AmharicLLMDatasetValidator._turns_from_messages(record)
        if turns is not None:
            assistant_turns = [c for r, c in turns if r in _ASSISTANT_ROLES]
            return assistant_turns[-1] if assistant_turns else ""
        if "output" in record:
            return record.get("output") or ""
        if "completion" in record:
            return record.get("completion") or ""
        return None

    # ---- Individual checks --------------------------------------------

    @staticmethod
    def find_missing_fields(records: Sequence[Record]) -> List[int]:
        """A record is missing required fields if it matches none of the
        recognized shapes, or matches instruction/prompt shape but the
        instruction/prompt itself is empty."""
        bad = []
        for i, rec in enumerate(records):
            turns = AmharicLLMDatasetValidator._turns_from_messages(rec)
            if turns is not None:
                continue  # shape presence checked in find_invalid_conversations
            if "instruction" in rec:
                if not str(rec.get("instruction") or "").strip():
                    bad.append(i)
                continue
            if "prompt" in rec:
                if not str(rec.get("prompt") or "").strip():
                    bad.append(i)
                continue
            # None of the recognized shapes present at all.
            bad.append(i)
        return bad

    @staticmethod
    def find_invalid_conversations(records: Sequence[Record]) -> List[int]:
        """Only applies to 'messages'/'conversations'-shaped records.
        Flags: empty turn list, a turn with no role or no content, and a
        conversation with no assistant turn at all (nothing for the model
        to learn to produce)."""
        bad = []
        for i, rec in enumerate(records):
            turns = AmharicLLMDatasetValidator._turns_from_messages(rec)
            if turns is None:
                continue
            if not turns:
                bad.append(i)
                continue
            has_assistant = False
            invalid = False
            for role, content in turns:
                if not role or role not in (_ASSISTANT_ROLES | _USER_ROLES | _SYSTEM_ROLES):
                    invalid = True
                if not str(content or "").strip():
                    invalid = True
                if role in _ASSISTANT_ROLES:
                    has_assistant = True
            if invalid or not has_assistant:
                bad.append(i)
        return bad

    def find_empty_responses(self, records: Sequence[Record]) -> List[int]:
        bad = []
        for i, rec in enumerate(records):
            resp = self._response_text(rec)
            if resp is not None and len(str(resp).strip()) < self.min_response_chars:
                bad.append(i)
        return bad

    def find_too_long(self, records: Sequence[Record], max_chars: Optional[int] = None) -> List[int]:
        limit = max_chars if max_chars is not None else self.max_chars
        return [i for i, rec in enumerate(records) if len(self._flat_text(rec)) > limit]

    def find_duplicates(self, records: Sequence[Record]) -> List[int]:
        """Delegates to AmharicCorpusProcessor.find_duplicates on the
        flattened text of each example."""
        flattened = [{"text": self._flat_text(rec)} for rec in records]
        return self._corpus.find_duplicates(flattened)

    def find_broken_unicode(self, records: Sequence[Record]) -> List[int]:
        return [
            i for i, rec in enumerate(records)
            if self._corpus.is_corrupt_text(self._flat_text(rec))
        ]

    @staticmethod
    def find_language_contamination(records: Sequence[Record], min_amharic_ratio: float = 0.3) -> List[int]:
        """
        Flags examples whose *response* contains almost no Amharic script
        at all -- i.e. a model trained on this example would learn to
        answer Amharic prompts in some other language. Only checked when
        the response is non-trivially long (short responses like "አዎ" or
        a bare number are legitimately all/mostly non-Amharic-script and
        shouldn't be flagged); `min_amharic_ratio` is the minimum fraction
        of *letter* characters in the response that must be Amharic.
        """
        bad = []
        validator = AmharicLLMDatasetValidator
        for i, rec in enumerate(records):
            resp = validator._response_text(rec)
            if not resp or len(str(resp).strip()) < 20:
                continue
            letters = [ch for ch in str(resp) if ch.isalpha()]
            if not letters:
                continue
            amharic_letters = sum(1 for ch in letters if AmharicLanguageDetector.is_amharic_char(ch))
            if (amharic_letters / len(letters)) < min_amharic_ratio:
                bad.append(i)
        return bad

    @staticmethod
    def find_poor_quality(records: Sequence[Record], min_unique_char_ratio: float = 0.15) -> List[int]:
        """
        Cheap gibberish/low-effort-sample heuristics on the response:
          - response is (after stripping whitespace/punctuation) an exact
            copy of the instruction/prompt -- the model would learn to
            echo, not answer.
          - response has very low character diversity (e.g. "ааааааааа"
            or a single repeated token spammed out) -- unique
            chars / total chars below `min_unique_char_ratio`.
        This is intentionally conservative (few, cheap heuristics) --
        it's a first pass to catch obviously broken generation/scraping
        artifacts, not a quality classifier.
        """
        import re as _re
        bad = []
        for i, rec in enumerate(records):
            resp = AmharicLLMDatasetValidator._response_text(rec)
            if not resp:
                continue
            resp_str = str(resp).strip()
            if not resp_str:
                continue

            prompt_str = str(
                rec.get("instruction") or rec.get("prompt") or ""
            ).strip()
            norm = lambda s: _re.sub(r"\s+", " ", s).strip().lower()
            if prompt_str and norm(resp_str) == norm(prompt_str):
                bad.append(i)
                continue

            if len(resp_str) >= 10:
                unique_ratio = len(set(resp_str)) / len(resp_str)
                if unique_ratio < min_unique_char_ratio:
                    bad.append(i)
        return bad

    def check_leakage(self, train_records: Sequence[Record],
                       eval_records: Sequence[Record]) -> List[int]:
        """
        Indices *into eval_records* whose flattened text (normalized,
        exact match) also appears anywhere in train_records -- i.e. the
        eval set isn't actually held out. Exact-match only (like
        find_duplicates); near-duplicate leakage across splits can be
        checked with AmharicCorpusProcessor.find_near_duplicates on the
        concatenation of both sets if needed.
        """
        train_texts = {
            self._corpus._normalize_for_comparison(self._flat_text(rec))
            for rec in train_records
        }
        train_texts.discard("")
        leaked = []
        for i, rec in enumerate(eval_records):
            key = self._corpus._normalize_for_comparison(self._flat_text(rec))
            if key and key in train_texts:
                leaked.append(i)
        return leaked

    # ---- Orchestration -------------------------------------------------

    def validate(self, records: Sequence[Record], max_chars: Optional[int] = None,
                 check_language_contamination: bool = False,
                 check_poor_quality: bool = True) -> LLMValidationReport:
        report = LLMValidationReport(total=len(records))
        report.missing_field_indices = self.find_missing_fields(records)
        report.invalid_conversation_indices = self.find_invalid_conversations(records)
        report.empty_response_indices = self.find_empty_responses(records)
        report.too_long_indices = self.find_too_long(records, max_chars)
        report.duplicate_indices = self.find_duplicates(records)
        report.broken_unicode_indices = self.find_broken_unicode(records)
        if check_language_contamination:
            report.language_contamination_indices = self.find_language_contamination(records)
        if check_poor_quality:
            report.poor_quality_indices = self.find_poor_quality(records)
        report.kept = report.total - len(report.bad_indices())
        return report

    def process(self, records: Sequence[Record], max_chars: Optional[int] = None,
                check_language_contamination: bool = False,
                check_poor_quality: bool = True) -> Tuple[List[Record], LLMValidationReport]:
        """Validate and return `(clean_records, report)` with every
        flagged example removed."""
        report = self.validate(
            records, max_chars=max_chars,
            check_language_contamination=check_language_contamination,
            check_poor_quality=check_poor_quality,
        )
        drop = report.bad_indices()
        clean = [rec for i, rec in enumerate(records) if i not in drop]
        report.kept = len(clean)
        return clean, report
