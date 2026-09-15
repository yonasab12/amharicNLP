"""
Dataset inspection: understand an Amharic dataset before training on it.

This is the read-only counterpart to validation. Validation answers "what
is wrong with this data"; inspection answers "what *is* this data" --
size, label distribution, text-length spread, how much of it is actually
Amharic script. Nothing here modifies or filters records.

The distinction matters in practice: a dataset can pass every validation
check and still be unfit to train on because 90% of it carries one label,
or because the median example is four characters long, or because a
"Amharic" dataset turns out to be 40% Latin script. Those are inspection
findings, not validation errors, and they're the ones most likely to
waste a training run.
"""

import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .resources.utils import AmharicLanguageDetector

Record = Dict[str, Any]


@dataclass
class DatasetReport:
    """Result of `inspect_dataset()`. Print it for a human-readable
    summary, or read the fields directly for programmatic checks."""

    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0

    columns: List[str] = field(default_factory=list)
    missing_per_column: Dict[str, int] = field(default_factory=dict)

    duplicate_samples: int = 0
    empty_text_samples: int = 0

    # Label statistics -- only populated when a label column was given.
    label_column: Optional[str] = None
    unique_labels: List[Any] = field(default_factory=list)
    label_distribution: Dict[Any, int] = field(default_factory=dict)
    label_imbalance_ratio: Optional[float] = None

    # Text statistics (characters).
    avg_char_length: float = 0.0
    min_char_length: int = 0
    max_char_length: int = 0
    median_char_length: float = 0.0

    # Text statistics (words, whitespace-split -- cheap and tokenizer-free,
    # so this number doesn't shift if you change tokenizers later).
    avg_word_count: float = 0.0
    min_word_count: int = 0
    max_word_count: int = 0
    median_word_count: float = 0.0

    # Amharic-specific.
    avg_amharic_ratio: float = 0.0
    mostly_amharic_samples: int = 0
    mostly_non_amharic_samples: int = 0
    ethiopic_punctuation_samples: int = 0

    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return self.summary()

    def summary(self) -> str:
        lines = [
            "=" * 58,
            "Dataset inspection report",
            "=" * 58,
            f"  total samples      : {self.total_samples}",
            f"  valid samples      : {self.valid_samples}",
            f"  invalid samples    : {self.invalid_samples}",
            f"  duplicates         : {self.duplicate_samples}",
            f"  empty text         : {self.empty_text_samples}",
            f"  columns            : {', '.join(self.columns) if self.columns else '(none)'}",
        ]

        missing = {k: v for k, v in self.missing_per_column.items() if v}
        if missing:
            lines.append("  missing values     :")
            for col, n in sorted(missing.items(), key=lambda kv: -kv[1]):
                lines.append(f"      {col}: {n}")

        if self.label_column:
            lines += [
                "-" * 58,
                f"  label column       : {self.label_column}",
                f"  unique labels      : {len(self.unique_labels)}",
            ]
            total = sum(self.label_distribution.values()) or 1
            for label, count in sorted(self.label_distribution.items(),
                                        key=lambda kv: -kv[1]):
                pct = 100.0 * count / total
                lines.append(f"      {label!r}: {count} ({pct:.1f}%)")
            if self.label_imbalance_ratio is not None:
                lines.append(f"  imbalance ratio    : {self.label_imbalance_ratio:.1f}x "
                              f"(most common / least common)")

        lines += [
            "-" * 58,
            f"  chars  avg/med/min/max : {self.avg_char_length:.1f} / "
            f"{self.median_char_length:.1f} / {self.min_char_length} / {self.max_char_length}",
            f"  words  avg/med/min/max : {self.avg_word_count:.1f} / "
            f"{self.median_word_count:.1f} / {self.min_word_count} / {self.max_word_count}",
            "-" * 58,
            f"  avg Amharic ratio      : {self.avg_amharic_ratio:.1%}",
            f"  mostly Amharic         : {self.mostly_amharic_samples}",
            f"  mostly non-Amharic     : {self.mostly_non_amharic_samples}",
            f"  uses Ethiopic punct.   : {self.ethiopic_punctuation_samples}",
        ]

        if self.warnings:
            lines += ["-" * 58, "  warnings:"]
            lines += [f"      ! {w}" for w in self.warnings]

        lines.append("=" * 58)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Report as a plain dict -- for logging to JSON, asserting in
        CI, or tracking dataset drift between versions."""
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "columns": list(self.columns),
            "missing_per_column": dict(self.missing_per_column),
            "duplicate_samples": self.duplicate_samples,
            "empty_text_samples": self.empty_text_samples,
            "label_column": self.label_column,
            "unique_labels": list(self.unique_labels),
            "label_distribution": dict(self.label_distribution),
            "label_imbalance_ratio": self.label_imbalance_ratio,
            "avg_char_length": self.avg_char_length,
            "min_char_length": self.min_char_length,
            "max_char_length": self.max_char_length,
            "median_char_length": self.median_char_length,
            "avg_word_count": self.avg_word_count,
            "min_word_count": self.min_word_count,
            "max_word_count": self.max_word_count,
            "median_word_count": self.median_word_count,
            "avg_amharic_ratio": self.avg_amharic_ratio,
            "mostly_amharic_samples": self.mostly_amharic_samples,
            "mostly_non_amharic_samples": self.mostly_non_amharic_samples,
            "ethiopic_punctuation_samples": self.ethiopic_punctuation_samples,
            "warnings": list(self.warnings),
        }


ETHIOPIC_PUNCTUATION = set("።፣፤፥፦፧፨፡")


def amharic_ratio(text: str) -> float:
    """Fraction of *letters* in `text` that are Ethiopic script. Letters
    only, so digits, punctuation, and whitespace don't dilute the signal
    -- "አዲስ አበባ 2016" should read as fully Amharic, not 75%."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    amharic = sum(1 for ch in letters if AmharicLanguageDetector.is_amharic_char(ch))
    return amharic / len(letters)


def inspect_dataset(
    records: Sequence[Record],
    text_column: str = "text",
    label_column: Optional[str] = None,
    min_length: int = 3,
    max_length: int = 100_000,
    amharic_threshold: float = 0.5,
) -> DatasetReport:
    """
    Produce a DatasetReport for `records` without modifying them.

    `min_length`/`max_length` are character bounds used only to count how
    many samples fall outside a sane range and to raise a warning -- they
    never filter. `amharic_threshold` is the letter-ratio above which a
    sample counts as "mostly Amharic".
    """
    report = DatasetReport(total_samples=len(records))
    if not records:
        report.warnings.append("Dataset is empty.")
        return report

    # Column inventory: union across records, so a ragged dataset reports
    # every column that appears anywhere, not just those in record 0.
    seen_cols: List[str] = []
    seen = set()
    for rec in records:
        for key in rec:
            if key not in seen:
                seen.add(key)
                seen_cols.append(key)
    report.columns = seen_cols
    report.missing_per_column = {col: 0 for col in seen_cols}

    char_lengths: List[int] = []
    word_counts: List[int] = []
    amharic_ratios: List[float] = []
    text_seen: Dict[str, int] = {}
    n_too_short = 0
    n_too_long = 0

    for rec in records:
        for col in seen_cols:
            value = rec.get(col, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                report.missing_per_column[col] += 1

        raw = rec.get(text_column, "")
        text = str(raw).strip() if raw is not None else ""

        if not text:
            report.empty_text_samples += 1
            report.invalid_samples += 1
            continue

        report.valid_samples += 1

        key = " ".join(unicodedata.normalize("NFC", text).split())
        text_seen[key] = text_seen.get(key, 0) + 1

        n_chars = len(text)
        char_lengths.append(n_chars)
        word_counts.append(len(text.split()))

        if n_chars < min_length:
            n_too_short += 1
        if n_chars > max_length:
            n_too_long += 1

        ratio = amharic_ratio(text)
        amharic_ratios.append(ratio)
        if ratio >= amharic_threshold:
            report.mostly_amharic_samples += 1
        else:
            report.mostly_non_amharic_samples += 1

        if any(ch in ETHIOPIC_PUNCTUATION for ch in text):
            report.ethiopic_punctuation_samples += 1

    report.duplicate_samples = sum(count - 1 for count in text_seen.values() if count > 1)

    if char_lengths:
        report.avg_char_length = sum(char_lengths) / len(char_lengths)
        report.min_char_length = min(char_lengths)
        report.max_char_length = max(char_lengths)
        report.median_char_length = statistics.median(char_lengths)
    if word_counts:
        report.avg_word_count = sum(word_counts) / len(word_counts)
        report.min_word_count = min(word_counts)
        report.max_word_count = max(word_counts)
        report.median_word_count = statistics.median(word_counts)
    if amharic_ratios:
        report.avg_amharic_ratio = sum(amharic_ratios) / len(amharic_ratios)

    # --- Labels -----------------------------------------------------------
    # Labels are discovered from the developer's data, never assumed. A
    # label set of {"positive","negative"} and one of {"ስፖርት","ፖለቲካ"} are
    # treated identically -- there is no privileged vocabulary here.
    if label_column:
        report.label_column = label_column
        distribution: Dict[Any, int] = {}
        for rec in records:
            label = rec.get(label_column, None)
            if label is None:
                continue
            key = label if isinstance(label, (str, int, float, bool)) else str(label)
            distribution[key] = distribution.get(key, 0) + 1
        report.label_distribution = distribution
        report.unique_labels = list(distribution.keys())
        if distribution:
            most, least = max(distribution.values()), min(distribution.values())
            report.label_imbalance_ratio = (most / least) if least else float("inf")

    # --- Warnings ----------------------------------------------------------
    # Warnings, not deletions: the developer decides what to do about them.
    if report.empty_text_samples:
        report.warnings.append(
            f"{report.empty_text_samples} sample(s) have empty/missing '{text_column}'."
        )
    if report.duplicate_samples:
        pct = 100.0 * report.duplicate_samples / max(report.total_samples, 1)
        report.warnings.append(
            f"{report.duplicate_samples} duplicate text(s) ({pct:.1f}% of dataset)."
        )
    if n_too_short:
        report.warnings.append(
            f"{n_too_short} sample(s) shorter than {min_length} characters."
        )
    if n_too_long:
        report.warnings.append(
            f"{n_too_long} sample(s) longer than {max_length} characters."
        )
    if report.mostly_non_amharic_samples:
        pct = 100.0 * report.mostly_non_amharic_samples / max(report.valid_samples, 1)
        if pct >= 10:
            report.warnings.append(
                f"{report.mostly_non_amharic_samples} sample(s) ({pct:.0f}%) are less than "
                f"{amharic_threshold:.0%} Amharic script -- check for language contamination."
            )
    if report.label_imbalance_ratio and report.label_imbalance_ratio >= 10:
        report.warnings.append(
            f"Label imbalance is {report.label_imbalance_ratio:.0f}x between the most and "
            f"least common label -- consider rebalancing or weighting the loss."
        )
    if report.label_column and len(report.unique_labels) == 1:
        report.warnings.append(
            f"Only one unique label ({report.unique_labels[0]!r}) -- a classifier "
            f"cannot learn anything from a single-class dataset."
        )
    if report.total_samples < 50:
        report.warnings.append(
            f"Only {report.total_samples} samples -- likely too few to train on, "
            f"though fine for testing a pipeline."
        )

    return report
