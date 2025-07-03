from resources.cleaner import AmharicCleaner
from resources.normalizer import AmharicNormalizer
from resources.lemmatizer import AmharicLemmatizer
from resources.stemmer import AmharicStemmer
from resources.utils import AmharicLanguageDetector
from resources.stopwrod import AmharicStopwordProcessor


# Sample Amharic text
sample_text = "በአገራችን ኢትዮጵያ �ላ ያሉ ተማሪዎች በትምህርት ላይ ትኩረት ማድረግ አለባቸው። 123 ቁጥር!"

# Initialize all processors
cleaner = AmharicCleaner()
normalizer = AmharicNormalizer()
lemmatizer = AmharicLemmatizer()
stemmer = AmharicStemmer()
language_detector = AmharicLanguageDetector()
stopword_processor = AmharicStopwordProcessor()

# Apply each processing step
print("Original Text:", sample_text)

# 1. Clean the text
cleaned_text = cleaner.clean(sample_text)
print("\nAfter Cleaning:", cleaned_text)

# 2. Normalize the text
normalized_text = normalizer.normalize(cleaned_text)
print("\nAfter Normalization:", normalized_text)

# 3. Language detection
is_amharic = language_detector.is_amharic(normalized_text)
print("\nIs Amharic?", is_amharic)

# 4. Stopword removal
without_stopwords = stopword_processor.remove_stopwords(normalized_text)
print("\nAfter Stopword Removal:", without_stopwords)

# 5. Lemmatization
lemmatized_text = [lemmatizer.lemmatize(word) for word in without_stopwords.split()]
print("\nAfter Lemmatization:", " ".join(lemmatized_text))

# 6. Stemming
stemmed_text = [stemmer.stem(word) for word in without_stopwords.split()]
print("\nAfter Stemming:", " ".join(stemmed_text))