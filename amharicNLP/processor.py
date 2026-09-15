"""
The Bring-Your-Own-Data / Bring-Your-Own-Model dataset processor.

`AmharicDatasetProcessor` is the single entry point a developer needs:
hand it their own data, in their own shape, with their own column names
and their own labels, tell it which NLP task they're preparing for, and
get back processed records ready for *their* model in *their* framework.

    from amharicNLP import AmharicDatasetProcessor

    processor = AmharicDatasetProcessor(task="classification")
    result = processor.process(
        data=my_dataframe,
        text_column="amharic_text",
        label_column="category",
    )
    result.to_dataframe()          # or .export("train.jsonl"), .to_hf_dataset(), ...

Three design rules run through everything here:

1. **Nothing is assumed.** Column names, label vocabularies, and the
   task itself all come from the developer. There is no built-in list of
   sentiment labels, no required schema, no "text"/"label" convention
   that breaks if your columns are called something else.

2. **Nothing destructive happens by default.** Normalization and cleaning
   are on because they're near-universally wanted and reversible in
   effect (they fix orthography, they don't discard meaning). Tokenizing,
   stemming, stopword removal and augmentation are *off* by default and
   must be switched on deliberately -- stemming an ASR transcript or an
   LLM pretraining corpus damages it, and a package that did that
   silently would be actively harmful.

3. **Nothing the developer didn't ask about is touched.** Metadata
   columns (speaker_id, source, domain, audio paths, annotation spans)
   pass through untouched. Processing writes to new columns by default
   rather than overwriting the originals, so the input is always
   recoverable.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import io_adapters
from .inspection import DatasetReport, inspect_dataset
from .resources.cleaner import AmharicCleaner
from .resources.normalizer import AmharicNormalizer
from .resources.sentence_segmenter import AmharicSentenceSegmenter
from .resources.stemmer import AmharicStemmer
from .resources.stopword import AmharicStopwordProcessor
from .resources.tokenizer import AmharicWordTokenizer

Record = Dict[str, Any]

#: Task names `AmharicDatasetProcessor` understands. "generic" applies
#: text processing with no task-specific structure checks -- the right
#: choice for a task not listed here, since an unknown task should mean
#: "no extra assumptions", not an error.
TASKS = (
    "generic",
    "classification",   # sentiment, topic, intent, spam, toxicity, ... any labels
    "ner",              # token classification / BIO / BIOES, any entity types
    "llm",              # pretraining, instruction tuning, chat, QA, generation
    "asr",              # speech-to-text: transcript side of audio/text pairs
    "tts",              # text-to-speech: text side of text/audio pairs
)

#: Tasks where aggressive text surgery corrupts the training signal, so
#: enabling it warrants a warning even when explicitly requested.
_TEXT_FIDELITY_TASKS = ("llm", "asr", "tts")


@dataclass
class ProcessingResult:
    """
    What `AmharicDatasetProcessor.process()` returns: the processed
    records plus everything learned about them on the way through.

    Convertible into whatever the developer's training stack wants --
    that's the whole point of the package ending here rather than at a
    model.
    """

    records: List[Record] = field(default_factory=list)
    text_column: str = "text"
    label_column: Optional[str] = None
    task: str = "generic"

    #: Labels discovered in the developer's data, in first-seen order.
    labels: List[Any] = field(default_factory=list)
    #: label -> contiguous integer id, for frameworks that want ints.
    label2id: Dict[Any, int] = field(default_factory=dict)
    id2label: Dict[int, Any] = field(default_factory=dict)

    report: Optional[DatasetReport] = None
    warnings: List[str] = field(default_factory=list)
    dropped: int = 0

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    # -- conversions to the developer's framework of choice --------------
    def to_records(self) -> List[Record]:
        return [dict(r) for r in self.records]

    def to_dict(self) -> Dict[str, List[Any]]:
        return io_adapters.export(self.records, "dict")

    def to_dataframe(self):
        """pandas DataFrame. Needs pandas installed."""
        return io_adapters.export(self.records, "dataframe")

    def to_hf_dataset(self):
        """datasets.Dataset. Needs `datasets` installed."""
        return io_adapters.export(self.records, "hf_dataset")

    def export(self, path: str, fmt: Optional[str] = None, **kwargs):
        """
        Write to disk. `fmt` is inferred from the file extension when not
        given (.jsonl -> jsonl, .csv -> csv, ...).
        """
        if fmt is None:
            import os
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            fmt = {"ndjson": "jsonl", "tab": "tsv", "text": "txt"}.get(ext, ext)
        return io_adapters.export(
            self.records, fmt, path=path, text_field=self.text_column, **kwargs
        )

    def split(self, ratios: Tuple[float, ...] = (0.8, 0.1, 0.1),
              seed: Optional[int] = 42, stratify: bool = False
              ) -> Dict[str, List[Record]]:
        """
        Split into named splits. Two ratios -> {"train","test"}; three ->
        {"train","validation","test"}.

        `stratify=True` keeps each label's proportion roughly equal across
        splits -- worth using on imbalanced classification data, where a
        random split can leave a rare label absent from validation
        entirely. Requires a label column.
        """
        import random

        if not (2 <= len(ratios) <= 3):
            raise ValueError("ratios must have 2 or 3 entries")
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

        names = ["train", "test"] if len(ratios) == 2 else ["train", "validation", "test"]
        rng = random.Random(seed)

        def _split_group(items: List[Record]) -> List[List[Record]]:
            items = list(items)
            rng.shuffle(items)
            n = len(items)
            out, start = [], 0
            for i, ratio in enumerate(ratios):
                end = n if i == len(ratios) - 1 else start + int(round(n * ratio))
                out.append(items[start:end])
                start = end
            return out

        if stratify:
            if not self.label_column:
                raise ValueError("stratify=True needs a label_column")
            groups: Dict[Any, List[Record]] = {}
            for rec in self.records:
                groups.setdefault(rec.get(self.label_column), []).append(rec)
            buckets: List[List[Record]] = [[] for _ in ratios]
            for group_items in groups.values():
                for bucket, part in zip(buckets, _split_group(group_items)):
                    bucket.extend(part)
            for bucket in buckets:
                rng.shuffle(bucket)
        else:
            buckets = _split_group(self.records)

        return dict(zip(names, buckets))

    def summary(self) -> str:
        lines = [
            f"ProcessingResult(task={self.task!r}, {len(self.records)} records)",
            f"  text column : {self.text_column}",
        ]
        if self.label_column:
            lines.append(f"  label column: {self.label_column} ({len(self.labels)} labels)")
        if self.dropped:
            lines.append(f"  dropped     : {self.dropped}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


class AmharicDatasetProcessor:
    """
    Task-aware Amharic dataset preparation, model- and framework-agnostic.

        # Sentiment with the developer's own labels
        proc = AmharicDatasetProcessor(task="classification")
        result = proc.process(df, text_column="text", label_column="sentiment")

        # NER with the developer's own entity types
        proc = AmharicDatasetProcessor(task="ner")
        result = proc.process(data, tokens_column="tokens", tags_column="ner_tags")

        # ASR transcripts -- normalized, never stemmed
        proc = AmharicDatasetProcessor(task="asr")
        result = proc.process(data, text_column="transcription",
                              metadata_columns=["audio", "speaker_id", "duration"])

    Constructor flags control the pipeline stages; every stage that can
    destroy information is off unless asked for.
    """

    def __init__(
        self,
        task: str = "generic",
        normalize: bool = True,
        clean: bool = True,
        tokenize: bool = False,
        stem: bool = False,
        remove_stopwords: bool = False,
        segment_sentences: bool = False,
        lowercase_latin: bool = False,
        preserve_latin: Optional[bool] = None,
        normalize_options: Optional[Dict[str, Any]] = None,
        drop_invalid: bool = False,
        min_length: int = 1,
        max_length: Optional[int] = None,
        inplace_columns: bool = False,
    ):
        """
        Args:
            task: one of TASKS. Controls task-specific validation and
                which structure the processor expects -- not which model
                you'll use afterwards.
            normalize / clean: on by default (orthographic fixes, noise
                removal). Safe for every task.
            tokenize / stem / remove_stopwords / segment_sentences: off by
                default. Each writes a *new* column and never replaces the
                text, so nothing is lost by enabling them.
            lowercase_latin: lowercase only Latin-script characters in
                mixed text. Ethiopic script has no case, so this is a
                no-op on pure Amharic and only affects embedded English.
            preserve_latin: whether cleaning keeps Latin-script text.
                `AmharicCleaner.remove_noise` strips non-Ethiopic
                characters, which is right for a pure-Amharic corpus and
                destructive for code-switched, translation, or
                transliterated data ("Hello" would become ""). Default
                None picks per task: preserved for llm/asr/tts (where the
                text is the training target and must stay faithful),
                stripped for classification/ner/generic (where non-Amharic
                noise is usually unwanted). Pass True/False to override.
            normalize_options: passed through to
                AmharicNormalizer.normalize_text -- e.g.
                {"url_mode": "placeholder", "collapse_repeated_chars": True}.
            drop_invalid: if True, records failing validation are removed
                and counted in `result.dropped`. Default False: report
                problems, keep the data, let the developer decide.
            min_length / max_length: character bounds used for validation
                warnings (and for dropping, when drop_invalid=True).
            inplace_columns: write results back over the source column
                instead of adding `<col>_clean` etc. Off by default so the
                original text stays recoverable.
        """
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}. Supported: {', '.join(TASKS)}")

        self.task = task
        self.do_normalize = normalize
        self.do_clean = clean
        self.do_tokenize = tokenize
        self.do_stem = stem
        self.do_remove_stopwords = remove_stopwords
        self.do_segment = segment_sentences
        self.lowercase_latin = lowercase_latin
        # Text-fidelity tasks keep Latin by default; see the constructor
        # docstring for why this is task-dependent rather than a fixed
        # default either way.
        self.preserve_latin = (
            task in _TEXT_FIDELITY_TASKS if preserve_latin is None else preserve_latin
        )
        self.normalize_options = dict(normalize_options or {})
        self.drop_invalid = drop_invalid
        self.min_length = min_length
        self.max_length = max_length
        self.inplace_columns = inplace_columns

        # Components are built lazily-ish: only what the configuration
        # actually uses, so a normalize-only processor doesn't construct a
        # stemmer or load the stopword list.
        self.cleaner = AmharicCleaner() if clean else None
        self.normalizer = AmharicNormalizer() if normalize else None
        self.segmenter = AmharicSentenceSegmenter() if segment_sentences else None
        needs_tokens = tokenize or stem or remove_stopwords
        self.tokenizer = AmharicWordTokenizer() if needs_tokens else None
        self.stemmer = AmharicStemmer() if stem else None
        self.stopwords = AmharicStopwordProcessor() if remove_stopwords else None

        self._config_warnings: List[str] = []
        if task in _TEXT_FIDELITY_TASKS and (stem or remove_stopwords):
            enabled = [n for n, on in (("stem", stem), ("remove_stopwords", remove_stopwords)) if on]
            self._config_warnings.append(
                f"task={task!r} with {'/'.join(enabled)} enabled: these alter the text "
                f"itself and normally degrade {task.upper()} training data. They're applied "
                f"to separate columns, so the original text is preserved -- but train on "
                f"the original unless you specifically want the altered version."
            )

    # -- text-level pipeline (usable standalone on a single string) -------

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Run the configured pipeline over one string. Returns a dict of
        every produced field ("text", and optionally "tokens", "stems",
        "sentences"), so callers can pick what they need.
        """
        if text is None:
            text = ""
        text = str(text)
        out: Dict[str, Any] = {}

        processed = text
        if self.do_clean and self.cleaner is not None:
            processed = self.cleaner.remove_html(processed)
            # keep_amharic=False removes emoji only, leaving Latin and
            # other scripts intact; keep_amharic=True additionally strips
            # everything non-Ethiopic.
            processed = self.cleaner.remove_noise(
                processed, keep_amharic=not self.preserve_latin
            )
        if self.do_normalize and self.normalizer is not None:
            processed = self.normalizer.normalize_text(processed, **self.normalize_options)
        if self.lowercase_latin:
            # Only ASCII letters are lowercased; Ethiopic is caseless and
            # str.lower() would be a no-op there anyway, but being explicit
            # documents the intent for mixed-script text.
            processed = "".join(ch.lower() if ch.isascii() and ch.isalpha() else ch
                                 for ch in processed)
        out["text"] = processed

        if self.do_segment and self.segmenter is not None:
            out["sentences"] = self.segmenter.split(processed)

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(processed)
            if self.do_remove_stopwords and self.stopwords is not None:
                tokens = [t for t in tokens if t not in self.stopwords.stop_words]
            if self.do_tokenize:
                out["tokens"] = tokens
            if self.do_stem and self.stemmer is not None:
                out["stems"] = [self.stemmer.stemaize(t) for t in tokens]

        return out

    # -- dataset-level entry points ---------------------------------------

    def inspect(self, data: Any, text_column: str = "text",
                label_column: Optional[str] = None, **kwargs) -> DatasetReport:
        """Inspect raw data without processing or modifying it."""
        records = io_adapters.to_records(data, text_field=text_column)
        return inspect_dataset(records, text_column=text_column,
                                label_column=label_column,
                                min_length=self.min_length,
                                max_length=self.max_length or 100_000, **kwargs)

    def validate(self, data: Any, text_column: str = "text",
                 label_column: Optional[str] = None, **kwargs) -> DatasetReport:
        """
        Alias of `inspect` kept as a separate name because the two read
        differently at a call site, and because validation is where
        task-specific structural checks are reported.
        """
        report = self.inspect(data, text_column, label_column, **kwargs)
        records = io_adapters.to_records(data, text_field=text_column)
        report.warnings.extend(self._task_warnings(records, text_column, label_column))
        return report

    def _task_warnings(self, records: Sequence[Record], text_column: str,
                        label_column: Optional[str]) -> List[str]:
        """Structural checks that only make sense for a given task. These
        are warnings, never deletions."""
        warnings: List[str] = []

        if self.task == "classification" and not label_column:
            warnings.append(
                "task='classification' but no label_column given -- the result will have "
                "no labels to train against."
            )

        if self.task == "ner":
            has_tokens = any("tokens" in r or "words" in r for r in records[:50])
            if not has_tokens:
                warnings.append(
                    "task='ner' but no token-level column detected -- pass "
                    "tokens_column=/tags_column= for token-classification data, or use "
                    "task='generic' for plain text."
                )

        if self.task in ("asr", "tts"):
            audio_keys = ("audio", "audio_path", "path", "file", "filename", "wav")
            has_audio = any(any(k in r for k in audio_keys) for r in records[:50])
            if not has_audio:
                warnings.append(
                    f"task={self.task!r} but no audio column detected (looked for "
                    f"{', '.join(audio_keys)}) -- pass it via metadata_columns so it's "
                    f"preserved alongside the processed text."
                )

        if self.task == "llm":
            llm_keys = ("instruction", "prompt", "messages", "conversations",
                        "input", "output", "completion", "response")
            has_structure = any(any(k in r for k in llm_keys) for r in records[:50])
            if has_structure:
                warnings.append(
                    "This looks like an instruction/chat dataset. AmharicLLMDatasetValidator "
                    "checks conversation structure, empty responses, and train/eval leakage "
                    "that this processor does not."
                )
        return warnings

    def process(
        self,
        data: Any,
        text_column: str = "text",
        label_column: Optional[str] = None,
        metadata_columns: Optional[Sequence[str]] = None,
        tokens_column: Optional[str] = None,
        tags_column: Optional[str] = None,
        output_column: Optional[str] = None,
        inspect: bool = True,
    ) -> ProcessingResult:
        """
        Process a dataset end to end.

        Args:
            data: anything `io_adapters.to_records` accepts.
            text_column: which column holds the Amharic text.
            label_column: optional; labels are read from the data, never
                assumed. Any label vocabulary is accepted as-is.
            metadata_columns: columns to explicitly preserve. Note that
                *all* columns are preserved regardless -- naming them here
                only asserts they must exist, and raises if one is missing,
                which is useful as a schema check in a pipeline.
            tokens_column / tags_column: for token-classification (NER)
                data, where text arrives pre-tokenized with per-token tags.
            output_column: where processed text goes. Defaults to
                `f"{text_column}_clean"`, or `text_column` itself when the
                processor was built with inplace_columns=True.
            inspect: attach a DatasetReport to the result (a second pass
                over the data; set False for very large datasets where you
                already know what's in them).
        """
        records = io_adapters.to_records(data, text_field=text_column)
        warnings: List[str] = list(self._config_warnings)

        if not records:
            return ProcessingResult(records=[], text_column=text_column,
                                     label_column=label_column, task=self.task,
                                     warnings=warnings + ["Input dataset is empty."])

        # Schema checks -- fail loudly on a missing column rather than
        # silently producing empty text, which is far harder to debug.
        present = set()
        for rec in records:
            present.update(rec.keys())

        if tokens_column and tags_column:
            for col in (tokens_column, tags_column):
                if col not in present:
                    raise KeyError(
                        f"Column {col!r} not found. Available columns: {sorted(present)}"
                    )
        elif text_column not in present:
            raise KeyError(
                f"text_column={text_column!r} not found. Available columns: "
                f"{sorted(present)}"
            )

        for col in (metadata_columns or ()):
            if col not in present:
                raise KeyError(
                    f"metadata_column {col!r} not found. Available columns: {sorted(present)}"
                )
        if label_column and label_column not in present:
            raise KeyError(
                f"label_column={label_column!r} not found. Available columns: {sorted(present)}"
            )

        warnings.extend(self._task_warnings(records, text_column, label_column))

        report = None
        if inspect:
            report = inspect_dataset(
                records, text_column=text_column, label_column=label_column,
                min_length=self.min_length, max_length=self.max_length or 100_000,
            )

        # --- token-classification path ---------------------------------
        # NER data is per-token, so the text pipeline would destroy the
        # token/tag alignment. Normalize each token individually instead,
        # leaving the tag sequence untouched and the same length.
        if tokens_column and tags_column:
            processed, dropped = self._process_token_classification(
                records, tokens_column, tags_column
            )
            labels = self._discover_labels_from_tags(processed, tags_column)
            result = ProcessingResult(
                records=processed, text_column=tokens_column, label_column=tags_column,
                task=self.task, labels=labels,
                label2id={l: i for i, l in enumerate(labels)},
                id2label=dict(enumerate(labels)),
                report=report, warnings=warnings, dropped=dropped,
            )
            return result

        # --- plain text path --------------------------------------------
        out_col = output_column or (text_column if self.inplace_columns
                                     else f"{text_column}_clean")

        processed_records: List[Record] = []
        dropped = 0
        for rec in records:
            new_rec = dict(rec)   # every original column survives
            produced = self.process_text(rec.get(text_column, ""))
            processed_text = produced["text"]

            if self.drop_invalid and not self._passes(processed_text):
                dropped += 1
                continue

            new_rec[out_col] = processed_text
            for extra_key, out_name in (
                ("tokens", "tokens"), ("stems", "stems"), ("sentences", "sentences"),
            ):
                if extra_key in produced:
                    new_rec[out_name] = produced[extra_key]
            processed_records.append(new_rec)

        labels: List[Any] = []
        if label_column:
            seen = set()
            for rec in processed_records:
                label = rec.get(label_column)
                if label is not None and not isinstance(label, (list, dict)) and label not in seen:
                    seen.add(label)
                    labels.append(label)

        return ProcessingResult(
            records=processed_records, text_column=out_col, label_column=label_column,
            task=self.task, labels=labels,
            label2id={l: i for i, l in enumerate(labels)},
            id2label=dict(enumerate(labels)),
            report=report, warnings=warnings, dropped=dropped,
        )

    # -- helpers -----------------------------------------------------------

    def _passes(self, text: str) -> bool:
        stripped = (text or "").strip()
        if len(stripped) < self.min_length:
            return False
        if self.max_length is not None and len(stripped) > self.max_length:
            return False
        return True

    def _process_token_classification(
        self, records: Sequence[Record], tokens_column: str, tags_column: str
    ) -> Tuple[List[Record], int]:
        """
        Normalize NER data token-by-token so tokens and tags stay aligned
        and equal in length. A token that normalizes to an empty string
        (e.g. a stray symbol) keeps its original form rather than being
        dropped -- dropping it would shift every subsequent tag.
        """
        processed: List[Record] = []
        dropped = 0

        for rec in records:
            tokens = rec.get(tokens_column)
            tags = rec.get(tags_column)

            if not isinstance(tokens, (list, tuple)) or not isinstance(tags, (list, tuple)):
                dropped += 1
                continue
            if len(tokens) != len(tags):
                # Misaligned annotation: keep it out of the processed set
                # rather than emitting silently-corrupt training data.
                dropped += 1
                continue

            new_tokens = []
            for tok in tokens:
                tok_str = str(tok)
                norm = tok_str
                if self.do_normalize and self.normalizer is not None:
                    norm = self.normalizer.normalize_text(
                        tok_str, normalize_punct_spacing=False, **self.normalize_options
                    ).strip()
                new_tokens.append(norm if norm else tok_str)

            new_rec = dict(rec)
            new_rec[tokens_column] = new_tokens
            new_rec[tags_column] = list(tags)
            processed.append(new_rec)

        return processed, dropped

    @staticmethod
    def _discover_labels_from_tags(records: Sequence[Record], tags_column: str) -> List[Any]:
        """Collect the entity-tag vocabulary from the developer's data,
        in first-seen order. Works for BIO, BIOES/BILOU, or any custom
        scheme -- no tag set is assumed."""
        labels: List[Any] = []
        seen = set()
        for rec in records:
            for tag in rec.get(tags_column, []):
                if tag not in seen:
                    seen.add(tag)
                    labels.append(tag)
        return labels
