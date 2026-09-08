"""Runnable demo. NOT imported automatically by the package anymore --
run it directly:  python -m amharicNLP.example.test
"""

from amharicNLP.resources.cleaner import AmharicCleaner
from amharicNLP.resources.normalizer import AmharicNormalizer
from amharicNLP.resources.lemmatizer import AmharicLemmatizer
from amharicNLP.resources.stemmer import AmharicStemmer
from amharicNLP.resources.stopwrod import AmharicStopwordProcessor
from amharicNLP.resources.tokenizer import AmharicWordTokenizer


def main():
    sample_text = (
        "በአገራችን ኢትዮጵያ <h1/> ላ ያሉ ተማሪዎች በትምህርት ላይ ትኩረት ማድረግ አለባቸው። "
        "123 ቁጥር! በላይ ዘለቀ ጀግና የኢትዮጵያ አርበኛ ነበር።"
    )

    cleaner = AmharicCleaner()
    normalizer = AmharicNormalizer()
    tokenizer = AmharicWordTokenizer()
    lemmatizer = AmharicLemmatizer()
    stemmer = AmharicStemmer()
    stopword_processor = AmharicStopwordProcessor()

    print("Original Text:", sample_text)

    cleaned = cleaner.remove_noise(cleaner.remove_html(sample_text))
    print("\nAfter Cleaning:", cleaned)

    normalized = normalizer.expand_abbreviations(
        normalizer.normalize_punctuation_spacing(
            normalizer.normalize_amharic_chars(cleaned)
        )
    )
    print("\nAfter Normalization:", normalized)

    tokens = tokenizer.tokenize(normalized)
    print("\nAfter Tokenization:", tokens)

    without_stopwords = stopword_processor.remove(normalized)
    print("\nAfter Stopword Removal:", without_stopwords)

    # NOTE: lemmatize/stem operate per word, not on a whole sentence.
    lemmatized = [lemmatizer.lemmatize(w) for w in without_stopwords]
    print("\nAfter Related-term Lookup:", lemmatized)

    stemmed = [stemmer.stemaize(w) for w in without_stopwords]
    print("\nAfter Stemming:", stemmed)


if __name__ == "__main__":
    main()
