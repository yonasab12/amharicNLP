"""
Simple text-augmentation utilities for Amharic (random deletion, synonym
replacement, random swap) -- for expanding a small labeled dataset (e.g.
sentiment/classification training data) with cheap, lexicon-free
perturbations.

NOTE: AmharicWordTokenizer only implements tokenize(), not a matching
detokenize(); an earlier version of this module called a
tokenizer.detokenize()/cleaner.remove_diacritics() that didn't exist
anywhere in the package, which meant augment_sentence() and this module's
own __main__ demo both raised AttributeError on the very first call. Fixed
below by joining tokens with a plain space (Amharic word tokenization
here doesn't need special detokenization rules -- there's no contraction
or punctuation-attachment logic to reverse) and by using the (now
implemented) AmharicCleaner.remove_diacritics().
"""

import random

from amharicNLP.resources.tokenizer import AmharicWordTokenizer
from amharicNLP.resources.cleaner import AmharicCleaner

__all__ = [
    "random_deletion",
    "synonym_replacement",
    "random_swap",
    "augment_sentence",
]

# Initialize class-based components
tokenizer = AmharicWordTokenizer()
cleaner = AmharicCleaner()

SYNONYMS = {
    "እንዴት": ["እንደምን", "በምን አይነት"],
    "ነህ": ["ነሽ", "ናችሁ"],
    "ደስ": ["ሐሰት", "ደስታ"],
    "ብሎኛል": ["አደሰኝ", "ደስ አለኝ"],
}


def _detokenize(words):
    """Join tokens back into a string. AmharicWordTokenizer's tokenize()
    doesn't do anything that needs undoing on the way back out (no
    contractions, no punctuation glued to the previous token) so a plain
    space join round-trips fine for this module's purposes."""
    return " ".join(words)


def random_deletion(words, p=0.2):
    if len(words) < 2:
        return words
    return [w for w in words if random.random() > p] or [random.choice(words)]


def synonym_replacement(words, n=1):
    new_words = words.copy()
    indices = [i for i, w in enumerate(new_words) if w in SYNONYMS]
    random.shuffle(indices)

    for i in indices[:min(n, len(indices))]:
        new_words[i] = random.choice(SYNONYMS[new_words[i]])
    return new_words


def random_swap(words, n=1):
    new_words = words.copy()
    for _ in range(min(n, len(words) // 2)):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words


def augment_sentence(sentence, aug_prob=0.3):
    words = tokenizer.tokenize(sentence)
    if random.random() < aug_prob:
        words = random_deletion(words)
    if random.random() < aug_prob:
        words = synonym_replacement(words, n=1)
    if random.random() < aug_prob:
        words = random_swap(words)
    return _detokenize(words)


if __name__ == "__main__":
    sentence = "እንዴት ነህ ደስ ብሎኛል"
    print("Original:", sentence)

    words = tokenizer.tokenize(sentence)

    print("Random Deletion:", _detokenize(random_deletion(words, p=0.3)))
    print("Synonym Replacement:", _detokenize(synonym_replacement(words, n=2)))
    print("Random Swap:", _detokenize(random_swap(words, n=2)))
    print("Remove Diacritics:", cleaner.remove_diacritics(sentence))
