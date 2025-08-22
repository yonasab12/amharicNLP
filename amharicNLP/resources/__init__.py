from .cleaner import AmharicCleaner
from .normalizer import AmharicNormalizer
from .lemmatizer import AmharicLemmatizer
from .stemmer import AmharicStemmer
from.utils import AmharicLanguageDetector
from .stopwrod import AmharicStopwordProcessor
from .amharic_stopword  import data
from.data_lemma import data as lemma_data


__all__ = ['AmharicCleaner', 'AmharicNormalizer', 'AmharicLemmatizer', 'AmharicStemmer','AmharicLanguageDetector', 'AmharicStopwordProcessor','data','lemma_data']