"""
Rule-based Amharic part-of-speech tagger.

No model, no training data, no external dependency -- this is closed-class
lexicon lookup (pronouns, conjunctions, auxiliaries, numbers, punctuation)
plus the same morphological affix patterns AmharicStemmerprocess already
knows about (verb prefixes like እየ-/ት-/ይ-, the አል...ም negation circumfix,
adjectival suffixes like -ኛ/-ዊ) to guess a tag for anything not in a
lexicon.

Tagset is a reduced Universal Dependencies set: NOUN, PROPN, VERB, AUX,
ADJ, ADV, PRON, DET, ADP, CCONJ, SCONJ, NUM, INTJ, PUNCT, X.

Honest limitations, up front:
  - Closed-class tags (PRON, CCONJ, SCONJ, AUX, NUM, PUNCT, most ADV/DET/
    INTJ) are lexicon lookups against a fixed, hand-curated word list.
    They're reliable for the words actually in the list and blind to
    everything else -- there is no generalization here.
  - VERB/ADJ detection is a morphological *guess*: matching a verb-typical
    prefix+suffix combination, a suffix-only perfective pattern, the
    negation circumfix (አል...ም and its አላ...ም allomorph), or an
    adjectival suffix. It will misfire on words that only coincidentally
    look like that pattern (the same trade-off documented in
    AmharicStemmerprocess) -- and it will still MISS real verbs whose
    ending isn't in the suffix list at all: tested against 30 real
    sentences, common past-tense forms like "አደረጉ" (they did), "ተጫወተ"
    (he played), "ዘነበ" (it rained) fall to the NOUN default because their
    endings (-ጉ, -ተ, -በ) aren't covered. Those endings were deliberately
    NOT added, because -ጉ/-ተ/-በ collide heavily with ordinary noun
    endings -- adding them fixed 2 verbs and broke 2 unrelated nouns
    ("ሰራተኞች" workers, "ጨዋታውን" the game[obj]) the first time this was
    tried. Widening this further needs either a labeled corpus to check
    against, or per-ending collision analysis this module doesn't have.
  - PROPN detection barely exists: Amharic script has no letter case, so
    there is no capitalization signal the way there is in English. This
    only fires for a tiny hand-curated set of well-known names/places.
  - Everything that doesn't match any rule falls back to NOUN, since NOUN
    is the most common open class -- this is a default, not a finding.

If you need POS tagging that's actually validated against a labeled
corpus (not just "seems reasonable on a handful of examples"), a
fine-tuned model (e.g. training a classifier on MasakhaPOS-style data) or
HornMorpho's real morphological analyzer will do much better than this.
This module exists because it was asked for as a from-scratch, dependency-
free, rule-based tagger -- treat its output as a rough first pass, not
ground truth.
"""

import re

# ---------------------------------------------------------------------------
# Closed-class lexicons (hand-curated, exact match only)
# ---------------------------------------------------------------------------

PRONOUNS = {
    # personal
    "እኔ", "አንተ", "አንቺ", "እሱ", "እሷ", "እኛ", "እናንተ", "እነሱ", "እነርሱ",
    # reflexive
    "ራሴ", "ራስህ", "ራስሽ", "ራሱ", "ራሷ", "ራሳችን", "ራሳችሁ", "ራሳቸው",
    # demonstrative (pronominal use)
    "ይህ", "ይሄ", "ያ", "እነዚህ", "እነዚያ",
    # interrogative
    "ማን", "ምን", "የት", "መቼ", "ለምን", "እንዴት", "የቱ", "የትኛው", "ስንት",
}

DETERMINERS = {
    "ሁሉ", "አንዳንድ", "ማንኛውም", "የትኛውም", "ብዙ", "ጥቂት", "እያንዳንዱ",
}

ADPOSITIONS = {
    # standalone adposition-like words (postpositions / relational nouns
    # that show up as separate tokens, as opposed to fused prefixes like
    # የ-/በ-/ከ-/ለ- which stay attached to their noun and are not split --
    # see the tokenizer's note on why we don't force-split those).
    "ጋር", "ውስጥ", "ላይ", "ስር", "ስለ", "ወደ", "ከ", "በ", "ለ", "የ", "አጠገብ",
    "መካከል", "በኩል", "ጋራ", "እንደ",
}

CCONJ = {"እና", "ወይም", "ግን", "እንዲሁም", "ወይ"}
SCONJ = {"ስለዚህ", "ምክንያቱም", "ቢሆንም", "እንደሆነ", "ከሆነ", "እንጂ", "ስለሆነም"}

AUX = {
    "ነው", "ናት", "ናቸው", "ነኝ", "ነህ", "ነሽ", "ናችሁ", "ነበር", "ነበሩ",
    "አለ", "አሉ", "የለም", "የላቸውም",
}

ADVERBS = {
    "አሁን", "ትናንት", "ነገ", "ዛሬ", "ገና", "ደግሞ", "በጣም", "እንደገና",
    "ሁልጊዜ", "አሁንም", "ቶሎ", "ቀስ", "በደንብ", "ወዲያውኑ", "ገና",
}

INTERJECTIONS = {"እሺ", "ዋው", "አይ", "ዋይ", "ኧረ", "ዋይኔ", "ዋሁ"}

# Cardinal number words (words, not just digits -- digits are handled by
# regex below). Not exhaustive (no attempt at compounds like "ሃያ አንድ").
NUMBER_WORDS = {
    "አንድ", "ሁለት", "ሶስት", "ሦስት", "አራት", "አምስት", "ስድስት", "ሰባት",
    "ስምንት", "ዘጠኝ", "አስር", "ሃያ", "ሰላሳ", "አርባ", "ሃምሳ", "ስልሳ",
    "ሰባ", "ሰማንያ", "ዘጠና", "መቶ", "ሺህ", "ሚሊዮን", "ቢሊዮን",
}

# A tiny, explicitly non-exhaustive set of well-known proper nouns. Amharic
# has no case marking, so this is the only signal available without a
# gazetteer. NOTE: multi-word names ("አዲስ አበባ") are deliberately NOT
# listed here -- tagging works on already-tokenized single words, and
# "አዲስ" alone is the ordinary adjective "new", so it can't be tagged
# PROPN without knowing it's followed by "አበባ". Multi-word entity
# detection needs a different approach (n-gram matching over the token
# list, done before/around tagging) that this module doesn't attempt.
KNOWN_PROPER_NOUNS = {
    "ኢትዮጵያ", "ኦሮሚያ", "አማራ", "ትግራይ",
    "አፍሪካ", "አውሮፓ", "አሜሪካ", "እንግሊዝ", "ቻይና",
}

# Common verb roots whose bare (3rd person masculine singular, past) form
# carries no prefix or suffix at all -- e.g. "ሄደ" (went), "መጣ" (came) --
# so the prefix+suffix pattern match in _looks_verbal can never catch
# them. Same trade-off as AmharicStemmerprocess.PROTECTED_WORDS: a fixed
# list patching a systematic blind spot, not a generalizable fix. Only
# high-frequency roots are listed; anything not here still falls through
# to the NOUN default.
BARE_VERB_ROOTS = {
    "ሄደ", "መጣ", "በላ", "ጠጣ", "ሰራ", "ሠራ", "ተኛ", "ወጣ", "ገባ",
    "አየ", "ሰማ", "ተናገረ", "ጻፈ", "ጠየቀ", "መለሰ", "ወሰደ", "ሰጠ",
    # short irregular/high-frequency inflected forms that fall below the
    # length guard on the general perfective-suffix check above (e.g.
    # "ሄዱ" is only 2 characters, well under the len>=5 threshold that
    # keeps that check from misfiring on short nouns)
    "ሄዱ", "መጡ", "በሉ", "ሰሩ", "ሠሩ", "ገቡ", "ወጡ",
}

_PUNCT_RE = re.compile(r"^[፠-፿.,;:!?()\[\]{}\"'…—\-]+$")
_DIGIT_RE = re.compile(r"^[0-9]+$")
_AMHARIC_NUMERAL_RE = re.compile(r"^[፩-፼]+$")

# Verb-typical affixes, reusing the same knowledge AmharicStemmerprocess
# uses for stripping -- here used only to *detect* a pattern, not to strip.
_VERB_PREFIXES = ("እየ", "ይ", "ት", "እን", "ስ", "ን", "አል")
_VERB_PREFIX_SUFFIXES = ("ኩ", "ሽ", "ህ", "ች", "ን", "ም", "ው", "ላችሁ", "ላቸው", "አለሁ")

# Perfective (past-tense) subject-agreement suffixes. Unlike imperfective/
# future verbs, Amharic perfective conjugation is suffix-only -- no subject
# prefix (ሰራ-ሁ/ሰራ-ሽ/ሰራ-ህ/ሰራ-ች/ሰራ-ን/ሰሩ, not ት-ሰራ or ይ-ሰራ). A prefix+suffix
# AND-check like _looks_verbal() structurally cannot match these, which
# excludes the single most common verb form in ordinary declarative
# sentences. Checked as a standalone suffix match instead (no prefix
# required), guarded by a longer minimum length since single-character
# suffixes here are the same over-eager risk documented in the stemmer.
_PERFECTIVE_SUFFIXES = ("ኩ", "ሽ", "ህ", "ች", "ን", "ችሁ", "ኡ", "ዱ", "ጡ", "ሁ")

# ...but ች and ን are exactly as common in NOUN morphology as in verb
# morphology: -ኞች/-ዎች/-ኦች is the standard noun plural, and -ውን/-ዋን is
# "the [direct object]" (definite + accusative). A bare endswith(ች)/
# endswith(ን) check misfires constantly on ordinary plural/object-marked
# nouns (e.g. "ሰራተኞች" workers, "ጨዋታውን" the game[obj]). Excluding these
# specific multi-character noun endings first keeps the single-character
# verb-suffix signal without also catching most nouns that merely end in
# the same last letter.
_NOUN_LIKE_ENDINGS = ("ኞች", "ዎች", "ኦች", "ውን", "ዋን")

# Negation circumfix አል...ም commonly surfaces as አላ...ም (with an epenthetic
# vowel) before roots starting with certain consonant clusters, e.g.
# አል+ነበብኩ -> አላነበብኩም ("didn't read"), አል+ሸነፈ -> አላሸነፈም ("didn't win").
# Matching only the bare "አል" prefix misses this very common variant.
_NEGATION_PREFIXES = ("አል", "አላ")

_ADJ_SUFFIXES = ("ዊ", "ኛ", "ማ")


class AmharicPOSTagger:
    """
    Rule-based POS tagger. No dependencies, no model.

        tagger = AmharicPOSTagger()
        tagger.tag_word("አልሄደም")          # 'VERB'
        tagger.tag(["እኔ", "ኢትዮጵያዊ", "ነኝ"])  # [('እኔ','PRON'), ('ኢትዮጵያዊ','ADJ'), ('ነኝ','AUX')]
    """

    def _looks_verbal(self, word: str) -> bool:
        has_prefix = word.startswith(_VERB_PREFIXES)
        has_suffix = word.endswith(_VERB_PREFIX_SUFFIXES)
        if has_prefix and has_suffix and len(word) >= 4:
            return True
        # Perfective: suffix-only, no prefix required (see note above).
        # Excludes common noun plural/object-marking endings first (see
        # _NOUN_LIKE_ENDINGS), then requires a longer minimum length than
        # the prefix+suffix case since this is a single, weaker signal.
        if word.endswith(_NOUN_LIKE_ENDINGS):
            return False
        if word.endswith(_PERFECTIVE_SUFFIXES) and len(word) >= 5:
            return True
        return False

    def _is_negated_verb(self, word: str) -> bool:
        return word.startswith(_NEGATION_PREFIXES) and word.endswith("ም") and len(word) >= 5

    def tag_word(self, word: str) -> str:
        if not word:
            return "X"

        if _PUNCT_RE.match(word):
            return "PUNCT"

        if _DIGIT_RE.match(word) or _AMHARIC_NUMERAL_RE.match(word) or word in NUMBER_WORDS:
            return "NUM"

        if word in KNOWN_PROPER_NOUNS:
            return "PROPN"

        if word in PRONOUNS:
            return "PRON"

        if word in DETERMINERS:
            return "DET"

        if word in AUX:
            return "AUX"

        if word in CCONJ:
            return "CCONJ"

        if word in SCONJ:
            return "SCONJ"

        if word in ADPOSITIONS:
            return "ADP"

        if word in ADVERBS:
            return "ADV"

        if word in INTERJECTIONS:
            return "INTJ"

        if self._is_negated_verb(word):
            return "VERB"

        if word in BARE_VERB_ROOTS:
            return "VERB"

        if self._looks_verbal(word):
            return "VERB"

        if word.endswith(_ADJ_SUFFIXES) and len(word) > 3:
            return "ADJ"

        return "NOUN"

    def tag(self, tokens):
        """tokens: list[str] (already tokenized). Returns list[(word, tag)]."""
        return [(w, self.tag_word(w)) for w in tokens]
