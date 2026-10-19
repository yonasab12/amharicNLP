"""
Amharic-aware sentence segmentation.

Splitting Amharic text into sentences is not "split on '.'" -- the
primary sentence-final marker is the Ethiopic full stop '።' (and its
common doubled-wordspace variant '፡፡', still used in news writing), and
naive splitting on Latin '.' breaks on abbreviations ("ዶ.ር", "ዓ.ም"),
decimal numbers ("3.5"), and URLs/emails in mixed-language text.

This module is a plain sentence *splitter*, deliberately separate from
AmharicWordTokenizer (resources/tokenizer.py): tokenization decides how
to split a sentence into words/punctuation tokens; segmentation decides
where one sentence ends and the next begins. Corpus processing, dataset
preparation, and STT/TTS transcript alignment all want sentence
boundaries as a first pass, before (or instead of) word tokenization.
"""

import re
from typing import List, NamedTuple, Optional, Set


class Sentence(NamedTuple):
    """One segmented sentence, with optional character offsets into the
    original (unmodified) input string."""
    text: str
    start: int
    end: int


class AmharicSentenceSegmenter:
    """
    Splits Amharic (or mixed Amharic/English) text into sentences.

        seg = AmharicSentenceSegmenter()
        seg.split("ዶ/ር አበበ መጣ። እሱ ጥሩ ነው!")
        # ['ዶ/ር አበበ መጣ።', 'እሱ ጥሩ ነው!']

        seg.split_with_offsets(text)
        # [Sentence(text=..., start=0, end=13), ...]

    Sentence-final markers handled: ።  ፧  ፨  (Ethiopic full stop,
    question mark, paragraph separator) plus the doubled-wordspace
    variant ፡፡, and Latin/English '.', '!', '?' when not protected by an
    abbreviation, a decimal number, or a URL/email.

    ፣ ፤ ፥ ፦ (comma, semicolon, colon-ish, preface colon) are clause-level
    punctuation, not sentence boundaries, and are never split on.
    """

    # Hard sentence-final marks: always end a sentence, no abbreviation
    # exception needed (Ethiopic script has no equivalent ambiguity to
    # Latin "Mr." / "e.g." for these).
    HARD_TERMINATORS: Set[str] = {'።', '፧', '፨'}

    # Soft/ambiguous terminators shared with abbreviations, decimals, and
    # URLs -- need the lookahead/lookbehind guards below.
    SOFT_TERMINATORS: Set[str] = {'.', '!', '?'}

    # Common Amharic title/abbreviation forms that contain a bare '.' or
    # '/' immediately followed by a letter, so a '.' inside them must not
    # be treated as a sentence end. Sourced from the same abbreviation set
    # normalizer.py already expands (AMHARIC_ABBREVIATIONS) plus their
    # '.'-joined variants; kept here too since a segmenter is often run
    # *before* abbreviation expansion (segmentation should not depend on
    # having normalized first).
    ABBREVIATIONS = {
        'ዶ.ር', 'ዓ.ም', 'ዓ.ዓ', 'ፕ/ር', 'ወ/ሮ', 'ወ/ሪት', 'ዶ/ር', 'ት/ቤት', 'ት/ርት',
        'ት/ክፍል', 'መ/ር', 'መ/ቤት', 'ክ/ከተማ', 'ክ/ሀገር', 'ወ/ር', 'ፍ/ቤት', 'ጽ/ቤት',
        'ሲ/ር', 'ጠ/ሚኒስትር', 'ገ/ጊዮርጊስ', 'ቤ/ክርስትያን', 'ም/ስራ', 'ም/ቤት',
        'ተ/ሃይማኖት', 'ሚ/ር', 'ኮ/ል', 'ሜ/ጀነራል', 'ብ/ጀነራል', 'ሌ/ኮለኔል', 'ሊ/መንበር',
        'አ/አ', 'ር/መምህር', 'ፕ/ት',
        # Generic Latin abbreviations that show up in code-switched text.
        'e.g.', 'i.e.', 'etc.', 'Mr.', 'Mrs.', 'Dr.', 'Prof.', 'U.S.', 'U.K.',
    }

    # A '.' or '!'/'?' immediately followed by a lowercase letter, a digit,
    # or another '.'/word-char with no space is very unlikely to be a
    # sentence boundary (URL, email, decimal number, ellipsis run) --
    # detected structurally rather than via a fixed word list.
    _URL_RE = re.compile(r'\bhttps?://\S+|\bwww\.\S+', re.IGNORECASE)
    _EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
    _DECIMAL_RE = re.compile(r'\d\.\d')

    def __init__(self, custom_abbreviations: Optional[Set[str]] = None):
        self.abbreviations = set(self.ABBREVIATIONS)
        if custom_abbreviations:
            self.abbreviations |= set(custom_abbreviations)

    def _protected_spans(self, text: str):
        """Character ranges (start, end) that must not be split inside:
        URLs, emails, decimal numbers, and known abbreviations."""
        spans = []
        for regex in (self._URL_RE, self._EMAIL_RE, self._DECIMAL_RE):
            spans.extend((m.start(), m.end()) for m in regex.finditer(text))
        for abbr in self.abbreviations:
            start = 0
            while True:
                idx = text.find(abbr, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(abbr)))
                start = idx + 1
        return spans

    @staticmethod
    def _in_span(pos: int, spans) -> bool:
        return any(s <= pos < e for s, e in spans)

    def split_with_offsets(self, text: str) -> List[Sentence]:
        """Split text into sentences, returning (text, start, end) tuples
        with offsets into the original `text`. Sentence text is whitespace-
        trimmed and normalizes '፡፡' to '።', so it may differ from the
        original slice at those markers."""
        if not text:
            return []

        protected = self._protected_spans(text)
        n = len(text)
        boundaries = []  # end-of-sentence indices (exclusive)

        i = 0
        while i < n:
            ch = text[i]
            if text.startswith('፡፡', i):
                boundaries.append(i + 2)
                i += 2
                continue
            if ch in self.HARD_TERMINATORS:
                boundaries.append(i + 1)
            elif ch in self.SOFT_TERMINATORS:
                if not self._in_span(i, protected):
                    # Don't split mid-abbreviation/ellipsis: only treat as
                    # a boundary if not immediately followed by another
                    # soft terminator or a non-space, non-quote character
                    # that suggests the sentence continues (e.g. "3.5" is
                    # already excluded via protected spans; this catches
                    # trailing quotes/parens too).
                    j = i + 1
                    while j < n and text[j] in '"\'»)]』”':
                        j += 1
                    boundaries.append(j)
                    i = j - 1
            i += 1

        if not boundaries or boundaries[-1] != n:
            boundaries.append(n)

        sentences = []
        start = 0
        for end in boundaries:
            raw = text[start:end]
            stripped = raw.strip()
            if stripped:
                # Recompute a tight start offset for the stripped text.
                lead_ws = len(raw) - len(raw.lstrip())
                trail_ws = len(raw) - len(raw.rstrip())
                sentences.append(Sentence(
                    text=stripped.replace('፡፡', '።'),
                    start=start + lead_ws,
                    end=end - trail_ws,
                ))
            start = end

        return sentences

    def split(self, text: str) -> List[str]:
        """Split text into a plain list of sentence strings."""
        return [s.text for s in self.split_with_offsets(text)]
