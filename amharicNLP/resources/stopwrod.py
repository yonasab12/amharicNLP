"""
Deprecated: this module's filename was a typo ("stopwrod" for "stopword").
The real implementation now lives in stopword.py. This file is kept as a
backward-compatible re-export only, for anyone already doing
`from amharicNLP.resources.stopwrod import AmharicStopwordProcessor` --
prefer importing from stopword.py (or `from amharicNLP import
AmharicStopwordProcessor`) going forward.
"""

from .stopword import AmharicStopwordProcessor

__all__ = ["AmharicStopwordProcessor"]
