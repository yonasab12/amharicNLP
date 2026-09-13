"""
Corpus loading, validation, and cleanup for Amharic datasets.

This is deliberately separate from `dataset_prep.py`
(AmharicDatasetPreprocessor), which is a thin bridge to Hugging Face
`datasets` for text-column *cleaning* (normalize/tokenize/stem a single
column, everything else passes through untouched). This module is the
step before that: getting arbitrary raw data -- a .txt file of one
sentence per line, a CSV export, a JSONL dump, a pandas DataFrame -- into
a uniform in-memory shape (`list[dict]`), and then validating it: which
rows are empty, which are exact or near-duplicates, which contain
corrupted/broken-Unicode text, which are missing required fields.

Every record is a plain `dict`. Whatever columns/fields a loader produces
are kept as-is (metadata like a speaker_id, a label, a duration_s column
all pass through untouched) -- this module only reasons about the
records' `text_field`, never silently drops other keys.

Dependency-light by design: TXT/CSV/JSON/JSONL loading uses only the
standard library. pandas and pyarrow (for DataFrame/Parquet) are imported
lazily, only if you actually call `load_dataframe`/`load_parquet`, so a
plain `pip install amharicNLP` still works for the txt/csv/json/jsonl case.
"""

import csv
import json
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .resources.utils import AmharicLanguageDetector

Record = Dict[str, Any]


@dataclass
class ValidationReport:
    """Result of AmharicCorpusProcessor.validate() / .process()."""
    total: int = 0
    empty_indices: List[int] = field(default_factory=list)
    duplicate_indices: List[int] = field(default_factory=list)
    near_duplicate_indices: List[int] = field(default_factory=list)
    corrupt_indices: List[int] = field(default_factory=list)
    missing_field_indices: Dict[str, List[int]] = field(default_factory=dict)
    kept: int = 0

    def summary(self) -> str:
        lines = [f"total records: {self.total}"]
        if self.empty_indices:
            lines.append(f"empty: {len(self.empty_indices)}")
        if self.duplicate_indices:
            lines.append(f"exact duplicates: {len(self.duplicate_indices)}")
        if self.near_duplicate_indices:
            lines.append(f"near-duplicates: {len(self.near_duplicate_indices)}")
        if self.corrupt_indices:
            lines.append(f"corrupt text: {len(self.corrupt_indices)}")
        for fname, idxs in self.missing_field_indices.items():
            if idxs:
                lines.append(f"missing '{fname}': {len(idxs)}")
        lines.append(f"kept: {self.kept}/{self.total}")
        return "\n".join(lines)


class AmharicCorpusProcessor:
    """
    Load raw data into `list[dict]` records, then validate/clean it.

        proc = AmharicCorpusProcessor()
        records = proc.load_jsonl("raw.jsonl")
        clean, report = proc.process(records, text_field="text")
        print(report.summary())

    Every `load_*` method returns `list[dict]`; every cleanup method takes
    and returns `list[dict]` (or `(list[dict], indices)` /
    `(list[dict], ValidationReport)` where noted) so loaders and cleanup
    steps compose freely, and results can be handed to
    `AmharicDatasetPreprocessor`/`AmharicLLMDatasetValidator` downstream.
    """

    # ---- Loading ---------------------------------------------------

    @staticmethod
    def load_txt(path: str, encoding: str = "utf-8", text_field: str = "text",
                  skip_blank: bool = True) -> List[Record]:
        """One record per line."""
        records: List[Record] = []
        with open(path, "r", encoding=encoding) as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if skip_blank and not line.strip():
                    continue
                records.append({text_field: line})
        return records

    @staticmethod
    def load_csv(path: str, encoding: str = "utf-8", delimiter: str = ",") -> List[Record]:
        """Each row -> a dict keyed by the header row."""
        with open(path, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return [dict(row) for row in reader]

    @staticmethod
    def load_json(path: str, encoding: str = "utf-8", records_key: Optional[str] = None) -> List[Record]:
        """
        Load a JSON file. Accepts either a top-level JSON array of
        objects, or a JSON object with the array under `records_key`
        (auto-detected among common names -- "data", "records",
        "examples" -- if `records_key` isn't given).
        """
        with open(path, "r", encoding=encoding) as f:
            payload = json.load(f)

        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            if records_key is not None:
                return payload[records_key]
            for key in ("data", "records", "examples"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
            raise ValueError(
                "load_json: top-level JSON object has no list under "
                "'data'/'records'/'examples' -- pass records_key explicitly."
            )

        raise ValueError(f"load_json: unsupported top-level JSON type {type(payload).__name__}")

    @staticmethod
    def load_jsonl(path: str, encoding: str = "utf-8", skip_blank: bool = True) -> List[Record]:
        """One JSON object per line (JSON Lines)."""
        records: List[Record] = []
        with open(path, "r", encoding=encoding) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    if skip_blank:
                        continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"load_jsonl: invalid JSON on line {lineno} of {path}: {exc}") from exc
        return records

    @staticmethod
    def load_dataframe(df) -> List[Record]:
        """Convert a pandas DataFrame to `list[dict]` (one dict per row)."""
        return df.to_dict(orient="records")

    @staticmethod
    def load_parquet(path: str, columns: Optional[Sequence[str]] = None) -> List[Record]:
        """Load a Parquet file. Needs pandas + pyarrow (or fastparquet)
        installed -- not a base dependency, since most non-LLM text
        corpora don't need columnar storage."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "load_parquet needs pandas (and pyarrow or fastparquet): "
                "pip install pandas pyarrow"
            ) from exc
        df = pd.read_parquet(path, columns=list(columns) if columns else None)
        return AmharicCorpusProcessor.load_dataframe(df)

    # ---- Validation / cleanup --------------------------------------

    @staticmethod
    def find_missing_fields(records: Sequence[Record], required_fields: Sequence[str]) -> Dict[str, List[int]]:
        """For each required field, indices of records where it's absent,
        None, or an empty/whitespace-only string."""
        missing: Dict[str, List[int]] = {f: [] for f in required_fields}
        for i, rec in enumerate(records):
            for f_name in required_fields:
                value = rec.get(f_name, None)
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing[f_name].append(i)
        return missing

    @staticmethod
    def find_empty(records: Sequence[Record], text_field: str = "text") -> List[int]:
        """Indices of records whose `text_field` is missing, None, or
        empty/whitespace-only after stripping."""
        return [
            i for i, rec in enumerate(records)
            if not str(rec.get(text_field, "") or "").strip()
        ]

    @staticmethod
    def _normalize_for_comparison(text: str) -> str:
        """NFC-normalize and collapse whitespace, purely for dedup
        comparison -- doesn't mutate the record's actual text."""
        return " ".join(unicodedata.normalize("NFC", text or "").split())

    def find_duplicates(self, records: Sequence[Record], text_field: str = "text",
                          keep: str = "first") -> List[int]:
        """Indices of exact-duplicate records (by normalized `text_field`
        value). `keep`: 'first' marks every occurrence after the first as
        a duplicate; 'none' marks every record that has any duplicate,
        including the first."""
        if keep not in ("first", "none"):
            raise ValueError("keep must be 'first' or 'none'")

        seen: Dict[str, int] = {}
        groups: Dict[str, List[int]] = {}
        for i, rec in enumerate(records):
            key = self._normalize_for_comparison(str(rec.get(text_field, "") or ""))
            if not key:
                continue
            groups.setdefault(key, []).append(i)

        duplicate_indices: List[int] = []
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            if keep == "first":
                duplicate_indices.extend(idxs[1:])
            else:
                duplicate_indices.extend(idxs)
        return sorted(duplicate_indices)

    def find_near_duplicates(self, records: Sequence[Record], text_field: str = "text",
                               threshold: float = 0.9,
                               exclude_indices: Optional[Set[int]] = None) -> List[int]:
        """
        Indices of near-duplicate records: for each record (in order),
        flag it if its normalized text is at least `threshold` similar
        (difflib.SequenceMatcher ratio, 0-1) to an already-kept earlier
        record. O(n^2) in the worst case -- fine for the small/medium
        corpora this is meant for (thousands of rows); for large-scale
        corpora, pre-filter with find_duplicates()/exclude_indices, or
        bucket by a cheap key (e.g. text length or a token-set hash)
        before calling this.

        `exclude_indices` (e.g. already-known exact duplicates or empty
        rows) are skipped entirely -- neither compared against nor
        reported.
        """
        if not (0.0 < threshold <= 1.0):
            raise ValueError("threshold must be in (0, 1]")
        exclude_indices = exclude_indices or set()

        kept_texts: List[Tuple[int, str]] = []
        near_dup_indices: List[int] = []
        for i, rec in enumerate(records):
            if i in exclude_indices:
                continue
            text = self._normalize_for_comparison(str(rec.get(text_field, "") or ""))
            if not text:
                continue
            is_near_dup = False
            for _, kept_text in kept_texts:
                if SequenceMatcher(None, text, kept_text).ratio() >= threshold:
                    is_near_dup = True
                    break
            if is_near_dup:
                near_dup_indices.append(i)
            else:
                kept_texts.append((i, text))
        return near_dup_indices

    @staticmethod
    def is_corrupt_text(text: str, amharic_threshold: Optional[float] = None) -> bool:
        """
        Heuristic corrupt/garbage-text detector. Flags text containing:
          - the Unicode replacement character '\\ufffd' (mojibake -- a
            sure sign of a decoding error somewhere upstream)
          - control characters other than tab/newline/carriage-return
            (null bytes, other C0/C1 controls -- OCR/encoding noise)
          - if `amharic_threshold` is given, text that is mostly neither
            Amharic nor plain ASCII (i.e. not "is_amharic_text" and not
            "looks like ordinary Latin/code-switched text") -- catches
            text that decoded into the wrong script entirely.
        """
        if not text:
            return False
        if "\ufffd" in text:
            return True
        if any((ord(ch) < 32 and ch not in "\t\n\r") or ord(ch) == 127 for ch in text):
            return True
        if amharic_threshold is not None:
            amharic_or_ascii = sum(
                1 for ch in text
                if AmharicLanguageDetector.is_amharic_char(ch) or ord(ch) < 128
            )
            if text and (amharic_or_ascii / len(text)) < (1 - amharic_threshold):
                return True
        return False

    def find_corrupt(self, records: Sequence[Record], text_field: str = "text",
                       amharic_threshold: Optional[float] = None) -> List[int]:
        return [
            i for i, rec in enumerate(records)
            if self.is_corrupt_text(str(rec.get(text_field, "") or ""), amharic_threshold)
        ]

    def validate(self, records: Sequence[Record], text_field: str = "text",
                 required_fields: Optional[Sequence[str]] = None,
                 near_duplicate_threshold: Optional[float] = None,
                 amharic_threshold: Optional[float] = None) -> ValidationReport:
        """Run every check and return a ValidationReport without modifying
        `records`. `near_duplicate_threshold=None` (default) skips the
        O(n^2) near-duplicate check -- opt in explicitly for corpora where
        it's worth the cost."""
        report = ValidationReport(total=len(records))
        report.empty_indices = self.find_empty(records, text_field)
        report.duplicate_indices = self.find_duplicates(records, text_field)
        report.corrupt_indices = self.find_corrupt(records, text_field, amharic_threshold)
        if required_fields:
            report.missing_field_indices = self.find_missing_fields(records, required_fields)
        if near_duplicate_threshold is not None:
            exclude = set(report.empty_indices) | set(report.duplicate_indices)
            report.near_duplicate_indices = self.find_near_duplicates(
                records, text_field, near_duplicate_threshold, exclude_indices=exclude,
            )
        bad = set(report.empty_indices) | set(report.duplicate_indices) | \
            set(report.corrupt_indices) | set(report.near_duplicate_indices)
        for idxs in report.missing_field_indices.values():
            bad |= set(idxs)
        report.kept = report.total - len(bad)
        return report

    def process(self, records: Sequence[Record], text_field: str = "text",
                required_fields: Optional[Sequence[str]] = None,
                remove_empty: bool = True, remove_duplicates: bool = True,
                remove_corrupt: bool = True,
                near_duplicate_threshold: Optional[float] = None,
                amharic_threshold: Optional[float] = None) -> Tuple[List[Record], ValidationReport]:
        """
        Validate `records` and return `(clean_records, report)`: the
        records with every issue you asked to remove filtered out (other
        fields/metadata on kept records are untouched), plus the full
        ValidationReport describing what was found and removed.
        Records failing `required_fields` are always removed (a record
        genuinely missing a required field isn't usable downstream
        regardless of the other flags).
        """
        report = self.validate(
            records, text_field=text_field, required_fields=required_fields,
            near_duplicate_threshold=near_duplicate_threshold,
            amharic_threshold=amharic_threshold,
        )

        drop: Set[int] = set()
        if remove_empty:
            drop |= set(report.empty_indices)
        if remove_duplicates:
            drop |= set(report.duplicate_indices)
        if remove_corrupt:
            drop |= set(report.corrupt_indices)
        drop |= set(report.near_duplicate_indices)
        for idxs in report.missing_field_indices.values():
            drop |= set(idxs)

        clean = [rec for i, rec in enumerate(records) if i not in drop]
        report.kept = len(clean)
        return clean, report
