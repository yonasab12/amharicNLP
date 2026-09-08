"""
Standalone re-export kept for backward compatibility.

The real implementation lives in stemmer.py, next to AmharicStemmerprocess,
which now also does the actual morphological normalization. See the note
on AmharicLemmatizer there: the data this class draws on is a related-term
table, not an inflection->lemma table, so this performs related-term
lookup rather than true lemmatization.
"""

from .stemmer import AmharicLemmatizer

__all__ = ["AmharicLemmatizer"]
