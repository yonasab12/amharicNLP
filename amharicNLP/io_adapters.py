"""
Bring-Your-Own-Data input adapters and the export system.

Whatever shape a developer's Amharic data arrives in -- a single string,
a list of strings, a list of dicts, a CSV/TSV/JSON/JSONL/TXT file, a
pandas DataFrame, a Hugging Face Dataset or DatasetDict -- `to_records()`
normalizes it to the one internal shape everything else in amharicNLP
works with: `list[dict]`, one dict per example, with the developer's own
column names preserved exactly as given.

Nothing here invents, renames, or drops columns. A record that arrives
with `speaker_id`, `source` and `domain` alongside its text leaves with
all three intact -- the processors only ever read the columns they're
told to read and write the ones they're told to write.

`export()` is the reverse: take processed records back out to whichever
format the developer's training pipeline wants. Everything past the
standard library (pandas, datasets, pyarrow) is imported lazily via
`_optional.require`, so none of it is a hard dependency.
"""

import csv
import io
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from ._optional import have, require

Record = Dict[str, Any]

#: Formats `export()` can write. "records"/"dataframe"/"hf_dataset" return
#: an in-memory object; the rest write a file and return its path.
EXPORT_FORMATS = (
    "records", "dict", "dataframe", "hf_dataset",
    "csv", "tsv", "json", "jsonl", "txt", "parquet",
)


# --------------------------------------------------------------------------
# Input  ->  list[dict]
# --------------------------------------------------------------------------

def _looks_like_hf_dataset(obj: Any) -> bool:
    """Duck-typed so we never import `datasets` just to check a type --
    a developer who doesn't use HF shouldn't pay an import for it."""
    return type(obj).__name__ in ("Dataset", "IterableDataset") and hasattr(obj, "features")


def _looks_like_hf_datasetdict(obj: Any) -> bool:
    return type(obj).__name__ in ("DatasetDict", "IterableDatasetDict")


def _looks_like_dataframe(obj: Any) -> bool:
    return type(obj).__name__ == "DataFrame" and hasattr(obj, "to_dict")


def from_file(path: str, text_field: str = "text", encoding: str = "utf-8",
              delimiter: Optional[str] = None) -> List[Record]:
    """
    Load a data file by extension: .csv, .tsv, .json, .jsonl/.ndjson,
    .txt, .parquet. Delegates to AmharicCorpusProcessor's loaders for the
    text formats rather than reimplementing them (those already handle
    the JSON-wrapped-under-a-key case, per-line error reporting, etc.).
    """
    from .corpus import AmharicCorpusProcessor

    proc = AmharicCorpusProcessor()
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return proc.load_csv(path, encoding=encoding, delimiter=delimiter or ",")
    if ext in (".tsv", ".tab"):
        return proc.load_csv(path, encoding=encoding, delimiter=delimiter or "\t")
    if ext == ".json":
        return proc.load_json(path, encoding=encoding)
    if ext in (".jsonl", ".ndjson"):
        return proc.load_jsonl(path, encoding=encoding)
    if ext in (".txt", ".text", ""):
        return proc.load_txt(path, encoding=encoding, text_field=text_field)
    if ext == ".parquet":
        return proc.load_parquet(path)

    raise ValueError(
        f"Unrecognized data file extension {ext!r} for {path!r}. "
        f"Supported: .csv .tsv .json .jsonl .txt .parquet -- or load it "
        f"yourself and pass the resulting list of dicts / DataFrame directly."
    )


def to_records(data: Any, text_field: str = "text",
               encoding: str = "utf-8") -> List[Record]:
    """
    Normalize any supported input to `list[dict]`.

    Accepts:
      - str                  -> a file path if it exists, otherwise treated
                                as a single text example
      - list[str]            -> one record per string, under `text_field`
      - list[dict]           -> returned as-is (copied, not mutated)
      - dict of lists        -> column-oriented, e.g. {"text": [...], "label": [...]}
      - dict (single record) -> one-element list
      - pandas DataFrame     -> one record per row
      - HF Dataset           -> one record per row
      - HF DatasetDict       -> rejected: split it first, or use
                                `to_records_dict()` to process every split
      - any other iterable of str/dict
    """
    if data is None:
        raise ValueError("to_records() got None -- pass a string, list, dict, DataFrame, or Dataset.")

    if _looks_like_hf_datasetdict(data):
        raise TypeError(
            "Got a DatasetDict (multiple splits). Either pass a single split "
            "(e.g. dataset['train']), or use to_records_dict() to get a "
            "{split_name: records} mapping back."
        )

    if isinstance(data, str):
        # A string that names an existing file is a path; otherwise it's
        # a single text example. Checking the filesystem (rather than
        # guessing from the presence of a "." or "/") avoids treating an
        # Amharic sentence that happens to contain a slash as a path.
        if os.path.exists(data):
            return from_file(data, text_field=text_field, encoding=encoding)
        return [{text_field: data}]

    if _looks_like_dataframe(data):
        return data.to_dict(orient="records")

    if _looks_like_hf_dataset(data):
        return [dict(row) for row in data]

    if isinstance(data, dict):
        # Column-oriented ({"text": [...], "label": [...]}) vs a single
        # record ({"text": "...", "label": "..."}). Column-oriented means
        # every value is a list of the same length.
        values = list(data.values())
        if values and all(isinstance(v, (list, tuple)) for v in values):
            lengths = {len(v) for v in values}
            if len(lengths) == 1:
                n = lengths.pop()
                return [{k: data[k][i] for k in data} for i in range(n)]
            raise ValueError(
                f"Column-oriented dict has mismatched column lengths: "
                f"{ {k: len(v) for k, v in data.items()} }"
            )
        return [dict(data)]

    if isinstance(data, Iterable):
        records: List[Record] = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                records.append(dict(item))
            elif isinstance(item, str):
                records.append({text_field: item})
            else:
                raise TypeError(
                    f"Item {i} of the input iterable is {type(item).__name__}; "
                    f"expected str or dict."
                )
        return records

    raise TypeError(
        f"Don't know how to read {type(data).__name__} as a dataset. Supported: "
        f"str (path or text), list[str], list[dict], dict, pandas DataFrame, "
        f"Hugging Face Dataset."
    )


def to_records_dict(data: Any, text_field: str = "text") -> Dict[str, List[Record]]:
    """
    Like `to_records`, but for multi-split inputs: a DatasetDict, or a
    plain dict-of-splits like {"train": [...], "test": [...]}. Returns
    {split_name: list[dict]}. Single-split inputs come back as
    {"train": [...]} so callers can treat both uniformly.
    """
    if _looks_like_hf_datasetdict(data):
        return {name: [dict(row) for row in split] for name, split in data.items()}

    if isinstance(data, dict) and data and all(
        isinstance(v, (list, tuple)) and (not v or isinstance(v[0], dict))
        for v in data.values()
    ):
        # dict-of-splits, not column-oriented (values are lists of dicts)
        return {name: [dict(r) for r in split] for name, split in data.items()}

    return {"train": to_records(data, text_field=text_field)}


# --------------------------------------------------------------------------
# list[dict]  ->  output
# --------------------------------------------------------------------------

def _fieldnames(records: Sequence[Record]) -> List[str]:
    """Union of keys across all records, in first-seen order -- so a
    ragged dataset (some records carrying an extra metadata column)
    exports every column rather than silently dropping the ones missing
    from record 0."""
    names: List[str] = []
    seen = set()
    for rec in records:
        for key in rec:
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def export(records: Sequence[Record], fmt: str = "records",
           path: Optional[str] = None, text_field: str = "text",
           encoding: str = "utf-8", **kwargs) -> Any:
    """
    Export processed records.

    In-memory formats (`path` ignored):
      "records"    -> list[dict] (the internal shape, copied)
      "dict"       -> column-oriented dict of lists
      "dataframe"  -> pandas DataFrame            (needs pandas)
      "hf_dataset" -> datasets.Dataset            (needs datasets)

    File formats (`path` required; returns the path written):
      "csv" "tsv" "json" "jsonl" "txt" "parquet"

    "txt" writes one record's `text_field` per line and drops every other
    column -- it's the only lossy format here, and appropriate mainly for
    LM pretraining corpora where the text is all that's wanted.
    """
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unknown export format {fmt!r}. Supported: {', '.join(EXPORT_FORMATS)}")

    records = [dict(r) for r in records]

    if fmt == "records":
        return records

    if fmt == "dict":
        names = _fieldnames(records)
        return {name: [rec.get(name) for rec in records] for name in names}

    if fmt == "dataframe":
        pd = require("pandas", "Exporting to a DataFrame")
        return pd.DataFrame(records)

    if fmt == "hf_dataset":
        datasets_mod = require("datasets", "Exporting to a Hugging Face Dataset")
        names = _fieldnames(records)
        columns = {name: [rec.get(name) for rec in records] for name in names}
        return datasets_mod.Dataset.from_dict(columns)

    # --- file formats ---
    if not path:
        raise ValueError(f"Export format {fmt!r} writes a file -- pass path=...")

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    if fmt in ("csv", "tsv"):
        delimiter = "," if fmt == "csv" else "\t"
        names = _fieldnames(records)
        with open(path, "w", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=names, delimiter=delimiter,
                                     extrasaction="ignore")
            writer.writeheader()
            for rec in records:
                # Lists (e.g. a tokens column) don't survive CSV as-is;
                # JSON-encode them so the column round-trips instead of
                # becoming Python's str([...]) repr.
                row = {
                    k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                    for k, v in rec.items()
                }
                writer.writerow(row)
        return path

    if fmt == "json":
        with open(path, "w", encoding=encoding) as f:
            json.dump(records, f, ensure_ascii=False, indent=kwargs.get("indent", 2))
        return path

    if fmt == "jsonl":
        with open(path, "w", encoding=encoding) as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    if fmt == "txt":
        with open(path, "w", encoding=encoding) as f:
            for rec in records:
                value = rec.get(text_field, "")
                if isinstance(value, list):
                    value = " ".join(str(v) for v in value)
                f.write(str(value).replace("\n", " ") + "\n")
        return path

    if fmt == "parquet":
        pd = require("pandas", "Exporting to Parquet")
        require("pyarrow", "Exporting to Parquet")
        pd.DataFrame(records).to_parquet(path, index=False)
        return path

    raise AssertionError(f"unreachable: {fmt}")  # pragma: no cover
