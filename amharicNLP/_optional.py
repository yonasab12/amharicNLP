"""
Optional-dependency handling.

amharicNLP's core (normalization, cleaning, tokenization, segmentation,
stemming, validation, the dataset processors) runs on the standard
library alone. Everything heavier -- pandas, PyTorch, TensorFlow,
Hugging Face transformers/datasets, scikit-learn, librosa -- is the
developer's choice to install, per-project, and is imported lazily at
the point of use rather than at import time.

That means `import amharicNLP` must never fail because someone hasn't
installed PyTorch, and a developer who only wants CSV in and JSONL out
should never be asked to install pandas. When a feature genuinely needs
an absent package, `require()` raises an ImportError naming the exact
pip command instead of a bare "No module named 'x'".

    from ._optional import require, have

    def to_dataframe(self):
        pd = require("pandas")          # raises with install hint if absent
        return pd.DataFrame(self.records)

    if have("pandas"):                   # branch without raising
        ...
"""

import importlib
from typing import Any, Optional

# Import name -> (pip install target, what it's needed for here).
# The pip name isn't always the import name (sklearn/scikit-learn,
# PIL/pillow), which is exactly why this mapping exists rather than
# interpolating the import name into the message.
_INSTALL_HINTS = {
    "pandas": ("pandas", "DataFrame input/export"),
    "numpy": ("numpy", "array operations"),
    "torch": ("torch", "PyTorch tensor/Dataset integration"),
    "tensorflow": ("tensorflow", "TensorFlow/Keras dataset integration"),
    "transformers": ("transformers", "Hugging Face model/tokenizer integration"),
    "datasets": ("datasets", "Hugging Face Dataset input/export"),
    "sklearn": ("scikit-learn", "scikit-learn integration"),
    "librosa": ("librosa", "audio loading/resampling"),
    "soundfile": ("soundfile", "audio file I/O"),
    "pyarrow": ("pyarrow", "Parquet input/export"),
    "tokenizers": ("tokenizers", "subword tokenizer training"),
}

# Extras declared in pyproject.toml, so the error can suggest the extra
# (which pulls the right pinned versions) rather than a bare package.
_EXTRA_FOR = {
    "datasets": "amharicNLP[datasets]",
    "transformers": "amharicNLP[neural]",
    "torch": "amharicNLP[neural]",
    "librosa": "amharicNLP[speech]",
    "soundfile": "amharicNLP[speech]",
}


class OptionalDependencyError(ImportError):
    """Raised when a feature needs a package the developer hasn't
    installed. Subclasses ImportError so existing `except ImportError`
    handlers keep working."""


def have(module_name: str) -> bool:
    """True if `module_name` can be imported. Use to branch on an
    optional capability without raising -- e.g. to offer a pandas
    return type only when pandas is present."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def require(module_name: str, feature: Optional[str] = None) -> Any:
    """
    Import and return `module_name`, or raise OptionalDependencyError
    with a copy-pasteable install command.

    `feature`: what the caller was trying to do, used in the message so
    the developer knows which call triggered it ("Parquet export needs
    pandas") rather than just that something, somewhere, wanted pandas.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pip_name, default_feature = _INSTALL_HINTS.get(
            module_name, (module_name, "this feature")
        )
        what = feature or default_feature
        lines = [
            f"{what} needs the optional package '{pip_name}', which isn't installed.",
            "",
            f"    pip install {pip_name}",
        ]
        if module_name in _EXTRA_FOR:
            lines += [
                "",
                "or, to get the version amharicNLP is tested against:",
                "",
                f"    pip install {_EXTRA_FOR[module_name]}",
            ]
        lines += [
            "",
            "amharicNLP's core preprocessing works without it -- this is an "
            "optional integration, not a required dependency.",
        ]
        raise OptionalDependencyError("\n".join(lines)) from exc
