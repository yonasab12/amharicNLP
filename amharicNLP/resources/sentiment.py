"""
Rule-based Amharic sentiment analysis. No model, no training data, no
dependency -- a hand-curated positive/negative word lexicon, matched
through the same stemmer the rest of the package uses (so inflected
forms like "ደስተኛ ነኝ" / "ደስተኛ ናት" both match the lexicon entry "ደስተኛ"),
plus negation handling via the pattern the stemmer/POS tagger already
know about (አል...ም / አላ...ም).

This is NOT the same thing as AmharicNeuralPipeline.sentiment() in
neural.py, which calls a real fine-tuned transformer model and will be
far more accurate on anything outside this lexicon. Use this when you
want zero dependencies and can live with lexicon coverage; use the
neural pipeline when you want real accuracy and can afford
`pip install amharicNLP[neural]`.

Honest limitations, up front (same discipline as the stemmer/POS tagger):
  - Coverage is exactly the words in POSITIVE_WORDS/NEGATIVE_WORDS below,
    reduced through the stemmer. Anything not in the lexicon (or a form
    the stemmer doesn't reduce to something in the lexicon) contributes
    nothing -- it is invisible to this analyzer, not neutral evidence.
  - Negation handling is pattern-based (አል...ም/አላ...ም) and only flips the
    sentiment of a NEGATED VERB if that verb form itself happens to also
    be a lexicon entry -- it does not detect "አልወደድኩትም" negating an
    unrelated noun's sentiment elsewhere in the sentence, and it does not
    handle negation words like "አይደለም" (is not) modifying an adjacent
    adjective ("ጥሩ አይደለም" = "not good" will score positive from "ጥሩ"
    alone unless "አይደለም" is treated as a standalone negator -- which it
    is, see NEGATORS below, but only for the word immediately preceding
    it in the same clause; it does not do real scope/dependency parsing).
  - No handling of sarcasm, intensifiers changing polarity, or context
    that a lexicon fundamentally cannot capture without real semantics.
  - This has NOT been benchmarked against a labeled sentiment dataset.
    It was checked against a small hand-built set of real sentences (see
    resources/tests/test_sentiment.py) -- treat it as a rough first pass.
"""

from .stemmer import AmharicStemProcessor

POSITIVE_WORDS = {
    "ጥሩ", "ደስ", "ደስተኛ", "ደስታ", "ምርጥ", "ቆንጆ", "አሪፍ", "ፍቅር", "ደግ",
    "ብልህ", "ስኬታማ", "ውብ", "ጣፋጭ", "አስደሳች", "እድለኛ", "ሰላም", "አዎንታዊ",
    "ጀግና", "ብርቱ", "ጥንካሬ", "ተስፋ", "እምነት", "አመሰግናለሁ", "አመስጋኝ",
    "ጎበዝ", "ንፁህ", "ጤናማ", "ደማቅ", "አዝናኝ", "አስደናቂ", "ተወዳጅ", "ውጤታማ",
}

NEGATIVE_WORDS = {
    "መጥፎ", "አስከፊ", "ክፉ", "አሳዛኝ", "ጨካኝ", "አስቀያሚ", "ደካማ", "ችግር",
    "ሀዘን", "ንዴት", "ፍርሃት", "አደገኛ", "ጎጂ", "አታካች", "ስንፍና", "ውድቀት",
    "ጭንቀት", "ብስጭት", "ጥላቻ", "ጦርነት", "ረሃብ", "በሽታ", "ሞት", "ስቃይ",
    "ፀብ", "ግጭት", "ስህተት", "ውሸት", "ስርቆት", "ወንጀል", "ጉዳት",
}

# Standalone words that negate whatever sentiment word directly precedes
# them in the token stream (e.g. "ጥሩ አይደለም" = "[it's] not good").
NEGATORS = {"አይደለም", "አይደለችም", "አይደሉም", "የለም", "አልነበረም"}


class AmharicSentimentAnalyzer:
    """
        analyzer = AmharicSentimentAnalyzer()
        analyzer.analyze("በጣም ጥሩ ፊልም ነበር።")
        # {'label': 'positive', 'score': 1, 'positive_matches': ['ጥሩ'], 'negative_matches': []}
    """

    def __init__(self):
        self._stemmer = AmharicStemProcessor()
        # Pre-stem the lexicons once so lookups are stem-to-stem.
        self._positive_stems = {self._stemmer.stem(w) for w in POSITIVE_WORDS} | POSITIVE_WORDS
        self._negative_stems = {self._stemmer.stem(w) for w in NEGATIVE_WORDS} | NEGATIVE_WORDS

    def _polarity_of(self, word: str) -> int:
        """1 positive, -1 negative, 0 no lexicon match."""
        if word in POSITIVE_WORDS or self._stemmer.stem(word) in self._positive_stems:
            return 1
        if word in NEGATIVE_WORDS or self._stemmer.stem(word) in self._negative_stems:
            return -1
        return 0

    def analyze(self, tokens_or_text) -> dict:
        """Accepts either a raw string (split on whitespace -- for
        punctuation-aware splitting, tokenize first and pass the token
        list) or an already-tokenized list of words."""
        tokens = tokens_or_text.split() if isinstance(tokens_or_text, str) else list(tokens_or_text)

        pos_matches, neg_matches = [], []
        score = 0

        for i, word in enumerate(tokens):
            # Negated verb (አል...ም / አላ...ም): if the negated form itself
            # is a lexicon entry (rare -- most verbs aren't sentiment
            # words), flip its polarity rather than double counting.
            if word.startswith(("አል", "አላ")) and word.endswith("ም"):
                stripped = word[2:-1] if word.startswith("አል") else word[3:-1]
                p = self._polarity_of(stripped)
                if p:
                    score -= p
                    (neg_matches if p > 0 else pos_matches).append(word)
                continue

            # "<word> አይደለም" etc: negator flips the immediately preceding
            # sentiment word instead of being scored on its own.
            if word in NEGATORS and i > 0:
                p = self._polarity_of(tokens[i - 1])
                if p:
                    score -= 2 * p  # cancel the earlier +1/-1 and flip it
                    prev_list = pos_matches if p > 0 else neg_matches
                    if tokens[i - 1] in prev_list:
                        prev_list.remove(tokens[i - 1])
                    (neg_matches if p > 0 else pos_matches).append(f"{tokens[i-1]} {word}")
                continue

            p = self._polarity_of(word)
            if p > 0:
                score += 1
                pos_matches.append(word)
            elif p < 0:
                score -= 1
                neg_matches.append(word)

        if score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "score": score,
            "positive_matches": pos_matches,
            "negative_matches": neg_matches,
        }
