"""
Rule-based Amharic stemmer.

Previous implementation only stripped the plural suffix "ዎች" and did a
linear substring search through a synonym table mislabeled as "lemma
data". This rewrite adds real affix stripping: Amharic verbs and nouns
carry chained prefixes (prepositions, relativizers, tense/aspect markers)
and suffixes (plural, possessive, object, negation) around a root, e.g.

    አልሄደም   = አል (neg) + ሄድ (root "go") + ም (neg)
    የቤታችን   = የ (of) + ቤት (house) + አችን (our)
    እየሄደች   = እየ (progressive) + ሄድ (root) + ች (3fs)

This module strips known prefix/suffix chains iteratively, longest match
first, guarding against over-stripping with a protected-word list and a
minimum remaining stem length.
"""

import os
from . import data_lemma

try:
    from .amharic_stopword import data as _stopword_data
except Exception:  # pragma: no cover - defensive import
    _stopword_data = []


# ----------------- Lemmatizer (synonym/related-term lookup) -----------------
class AmharicLemmatizer:
    """
    NOTE: the underlying `data_lemma` table maps a term to a list of
    *semantically related words / synonyms*, not inflected forms of the
    same word (e.g. "አጭር" (short) -> ["ድንክ", "ድውይ", "ከረከንዳ"]). It is
    therefore used here as a related-term lookup, not true morphological
    lemmatization. Real lemmatization is handled by AmharicStemmerprocess
    below via rule-based affix stripping.
    """

    def __init__(self, wordnet_data=None):
        self.wordnet_data = wordnet_data or data_lemma.data
        # Build an O(1) lookup: example word -> lemma, instead of the
        # previous O(n) linear scan over every entry for every word.
        self._index = {}
        for entry in self.wordnet_data:
            for example in entry.get("example", []):
                self._index.setdefault(example, entry["lemma"])

    def lemmatize(self, word):
        return self._index.get(word, word)


# ----------------- Stemmer Process -----------------
class AmharicStemmerprocess:
    # Words that must never be stripped, either because stripping would
    # destroy their meaning or because they only coincidentally start/end
    # with an affix-like sequence.
    PROTECTED_WORDS = {
        "ኢትዮጵያ", "አዲስ", "አበባ", "ሰላም", "ፍቅር", "ቤት", "ስራ", "ልጅ",
        "እኔ", "እኛ", "አንተ", "አንቺ", "እሱ", "እሷ", "እናንተ", "እነሱ",
        "አማርኛ", "ኦሮሚያ", "አፍሪካ", "አውሮፓ", "አሜሪካ",
        "ትምህርት", "ትዕግስት", "ትርፍ", "ትእዛዝ", "ትውልድ",
    }

    # Verbal / nominal prefixes, longest first so e.g. "እንደ" is tried
    # before "እ". Multi-character prefixes are safe to strip fairly
    # aggressively; single-character ones require a longer remaining stem.
    PREFIXES = [
        "እንደአስ", "እያንዳንዳ",
        "እንደተ", "እንደገ", "የሚያ", "የማይ",
        "እንደ", "እንዲ", "የሚ", "እየ", "ስለ", "እስከ", "ወደ", "በኩል", "አስ", "ባለ", "በየ",
        "እን", "ት", "ይ", "ስ", "ብ", "ሲ", "ቢ",
        "የ", "በ", "ከ", "ለ",
        "አ", "እ", "መ",
    ]

    # Nominal/verbal suffixes: possessive & plural clusters first (longest
    # match), then single-character person/number/case/negation markers.
    SUFFIXES = [
        "ዎቻችን", "ዎቻችሁ", "ዎቻቸው",
        "አቸውን", "አችንን",
        "ኦቻቸው",
        "ዎች", "ኦች", "ኡት", "ኧት", "አቸው", "አችን", "አችሁ", "አቸን", "ኧች",
        "ኝነት", "ነት",
        "ኝ", "ህ", "ሽ", "ው", "ዋ", "ን", "ት", "ም", "ች", "ኧ", "ላ", "ማ", "ያ",
        "ቶ", "ኦ", "ይ", "ሊ", "ስ",
    ]

    # Amharic alphabet variations for normalization (orders within a
    # group index the 7 "forms" of each consonant: 0=ə/ base,1=u,2=i,
    # 3=a,4=e,5=(sixth/bare),6=o)
    amharic_alphabet = [
        {"base": "ሀ", "variations": ["ሀ", "ሁ", "ሂ", "ሃ", "ሄ", "ህ", "ሆ"]},
        {"base": "ለ", "variations": ["ለ", "ሉ", "ሊ", "ላ", "ሌ", "ል", "ሎ"]},
        {"base": "ሐ", "variations": ["ሐ", "ሑ", "ሒ", "ሓ", "ሔ", "ሕ", "ሖ"]},
        {"base": "መ", "variations": ["መ", "ሙ", "ሚ", "ማ", "ሜ", "ም", "ሞ"]},
        {"base": "ሠ", "variations": ["ሠ", "ሡ", "ሢ", "ሣ", "ሤ", "ሥ", "ሦ"]},
        {"base": "ረ", "variations": ["ረ", "ሩ", "ሪ", "ራ", "ሬ", "ር", "ሮ"]},
        {"base": "ሰ", "variations": ["ሰ", "ሱ", "ሲ", "ሳ", "ሴ", "ስ", "ሶ"]},
        {"base": "ሸ", "variations": ["ሸ", "ሹ", "ሺ", "ሻ", "ሼ", "ሽ", "ሾ"]},
        {"base": "ቀ", "variations": ["ቀ", "ቁ", "ቂ", "ቃ", "ቄ", "ቅ", "ቆ"]},
        {"base": "በ", "variations": ["በ", "ቡ", "ቢ", "ባ", "ቤ", "ብ", "ቦ"]},
        {"base": "ተ", "variations": ["ተ", "ቱ", "ቲ", "ታ", "ቴ", "ት", "ቶ"]},
        {"base": "ቸ", "variations": ["ቸ", "ቹ", "ቺ", "ቻ", "ቼ", "ች", "ቾ"]},
        {"base": "ኀ", "variations": ["ኀ", "ኁ", "ኂ", "ኃ", "ኄ", "ኅ", "ኆ"]},
        {"base": "ነ", "variations": ["ነ", "ኑ", "ኒ", "ና", "ኔ", "ን", "ኖ"]},
        {"base": "ኘ", "variations": ["ኘ", "ኙ", "ኚ", "ኛ", "ኜ", "ኝ", "ኞ"]},
        {"base": "አ", "variations": ["አ", "ኡ", "ኢ", "ኣ", "ኤ", "እ", "ኦ"]},
        {"base": "ከ", "variations": ["ከ", "ኩ", "ኪ", "ካ", "ኬ", "ክ", "ኮ"]},
        {"base": "ኸ", "variations": ["ኸ", "ኹ", "ኺ", "ኻ", "ኼ", "ኽ", "ኾ"]},
        {"base": "ወ", "variations": ["ወ", "ዉ", "ዊ", "ዋ", "ዌ", "ው", "ዎ"]},
        {"base": "ዐ", "variations": ["ዐ", "ዑ", "ዒ", "ዓ", "ዔ", "ዕ", "ዖ"]},
        {"base": "ዘ", "variations": ["ዘ", "ዙ", "ዚ", "ዛ", "ዜ", "ዝ", "ዞ"]},
        {"base": "ዠ", "variations": ["ዠ", "ዡ", "ዢ", "ዣ", "ዤ", "ዥ", "ዦ"]},
        {"base": "የ", "variations": ["የ", "ዩ", "ዪ", "ያ", "ዬ", "ይ", "ዮ"]},
        {"base": "ደ", "variations": ["ደ", "ዱ", "ዲ", "ዳ", "ዴ", "ድ", "ዶ"]},
        {"base": "ጀ", "variations": ["ጀ", "ጁ", "ጂ", "ጃ", "ጄ", "ጅ", "ጆ"]},
        {"base": "ገ", "variations": ["ገ", "ጉ", "ጊ", "ጋ", "ጌ", "ግ", "ጎ"]},
        {"base": "ጠ", "variations": ["ጠ", "ጡ", "ጢ", "ጣ", "ጤ", "ጥ", "ጦ"]},
        {"base": "ጨ", "variations": ["ጨ", "ጩ", "ጪ", "ጫ", "ጬ", "ጭ", "ጮ"]},
        {"base": "ጰ", "variations": ["ጰ", "ጱ", "ጲ", "ጳ", "ጴ", "ጵ", "ጶ"]},
        {"base": "ጸ", "variations": ["ጸ", "ጹ", "ጺ", "ጻ", "ጼ", "ጽ", "ጾ"]},
        {"base": "ፀ", "variations": ["ፀ", "ፁ", "ፂ", "ፃ", "ፄ", "ፅ", "ፆ"]},
        {"base": "ፈ", "variations": ["ፈ", "ፉ", "ፊ", "ፋ", "ፌ", "ፍ", "ፎ"]},
        {"base": "ፐ", "variations": ["ፐ", "ፑ", "ፒ", "ፓ", "ፔ", "ፕ", "ፖ"]},
        {"base": "ቨ", "variations": ["ቨ", "ቩ", "ቪ", "ቫ", "ቬ", "ቭ", "ቮ"]},
    ]

    def __init__(self, min_stem_len: int = 2, max_passes: int = 1):
        self.min_stem_len = min_stem_len
        self.max_passes = max_passes

        self.normalize_map = {}
        for group in self.amharic_alphabet:
            for var in group["variations"]:
                self.normalize_map[var] = group["base"]

        # Allowed index rule (1,2,4) kept from the original ዎች-suffix rule
        self.allowed_indexes = {1, 2, 4}

        # Sort affix lists longest-first once, so matching is deterministic.
        self._prefixes = sorted(set(self.PREFIXES), key=len, reverse=True)
        self._suffixes = sorted(set(self.SUFFIXES), key=len, reverse=True)

    def _clean_woche_suffix(self, word):
        """Remove ዎች only if the preceding consonant form is in the
        allowed set (kept from the original heuristic)."""
        if not word.endswith("ዎች"):
            return word
        base = word[:-2]
        if not base:
            return word
        last_char = base[-1]
        for group in self.amharic_alphabet:
            if last_char in group["variations"]:
                idx = group["variations"].index(last_char)
                if idx in self.allowed_indexes:
                    return base
        return word

    def normalize(self, word):
        return ''.join(self.normalize_map.get(ch, ch) for ch in word)

    def _strip_negation(self, word):
        """Strip the circumfix negation marker አል...ም (e.g. አልሄደም -> ሄደ)."""
        if word.startswith("አል") and word.endswith("ም") and len(word) - 3 >= self.min_stem_len:
            return word[2:-1]
        return word

    def _strip_one_suffix(self, word):
        for suf in self._suffixes:
            if word.endswith(suf) and len(word) - len(suf) >= self.min_stem_len:
                return word[: -len(suf)]
        return word

    def _strip_one_prefix(self, word):
        for pre in self._prefixes:
            if word.startswith(pre) and len(word) - len(pre) >= self.min_stem_len:
                return word[len(pre):]
        return word

    def stem(self, word):
        """Strip known Amharic prefixes/suffixes down to an approximate root."""
        if isinstance(word, list):
            return [self.stem(w) for w in word]

        if not word or word in self.PROTECTED_WORDS or len(word) <= 2:
            return word

        result = word
        result = self._strip_negation(result)
        result = self._clean_woche_suffix(result)

        # max_passes defaults to 1 (single prefix-strip + single
        # suffix-strip). A higher value lets one weak single-character
        # affix match chain into a second weak match on the already-
        # stripped result -- e.g. "ጠቅላይ" (senior/premier) stripping
        # suffix "ይ" then, on the next pass, suffix "ላ" too, collapsing
        # it to "ጠቅ". Neither strip alone is unreasonable; compounding
        # two single-character guesses is what does the damage. Raise
        # this only if you've checked it against real sentences, not
        # just isolated test words.
        for _ in range(self.max_passes):
            before = result
            result = self._strip_one_prefix(result)
            result = self._strip_one_suffix(result)
            if result == before or len(result) <= self.min_stem_len:
                break

        # NOTE: we deliberately do NOT run the stripped stem through
        # self.normalize() here. That map collapses every vowel-order form
        # of a consonant to its bare/6th-order form (e.g. ሄ/ሂ/ሁ/ሆ -> ህ),
        # which is real grammatical information, not spelling noise -- the
        # original code did this and it silently destroyed roots (e.g.
        # "ሄደ" -> "ህደ"). normalize() is kept as a public method for callers
        # who explicitly want that (heavier) collapsing.
        return result if len(result) >= self.min_stem_len else word


# ----------------- Combined Rule: Lemmatize → Stem -----------------
class AmharicStemmer:
    def __init__(self, wordnet_data=None, min_stem_len: int = 2, max_passes: int = 1):
        self.lemmatizer = AmharicLemmatizer(wordnet_data)
        self.stemmer = AmharicStemmerprocess(min_stem_len=min_stem_len, max_passes=max_passes)

    def stemaize(self, word):
        """
        Rule-based affix stripping (the actual morphological
        normalization). This intentionally does NOT fall back to the
        related-term/synonym lookup: swapping a word for a synonym is a
        different operation from stemming and produced surprising results
        (e.g. "ሄደ" -> "ተበተነ", an unrelated word). Use
        AmharicStemmer.related_terms() explicitly if that's what you want.
        """
        return self.stemmer.stem(word)

    def related_terms(self, word):
        """Explicit access to the synonym/related-term table (see the
        note on AmharicLemmatizer). Not part of stemming."""
        return self.lemmatizer.lemmatize(word)

    def process_text(self, text):
        """Remove stopwords, then apply the Stem→related-term rule."""
        tokens = text.split()
        cleaned = [w for w in tokens if w not in _stopword_data]
        return [self.stemaize(w) for w in cleaned]


# ----------------- Example Usage -----------------
if __name__ == "__main__":
    stemmer = AmharicStemmer()

    test_words = [
        "አልሄደም", "የቤታችን", "እየሄደች", "በትምህርት", "ተማሪዎች",
        "ኢትዮጵያዊ", "ኮዎች", "ኢትዮጵያዎች", "ሰላም",
    ]

    print("Rule-based stem results:\n")
    for word in test_words:
        result = stemmer.stemaize(word)
        print(f"{word}  ->  {result}")
