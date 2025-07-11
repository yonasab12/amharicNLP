

from amharicNLP.resources.cleaner import AmharicCleaner
from amharicNLP.resources.normalizer import AmharicNormalizer
from amharicNLP.resources.lemmatizer import AmharicLemmatizer
from amharicNLP.resources.stemmer import AmharicStemmer           
from amharicNLP.resources.stopwrod import AmharicStopwordProcessor
from amharicNLP.resources.tokenizer import AmharicWordTokenizer

# Sample Amharic text
sample_text = "በአገራችን ኢትዮጵያ <h1/> �ላ ያሉ ተማሪዎች በትምህርት ላይ ትኩረት ማድረግ አለባቸው። 123 ቁጥር! በላይ ዘለቀ ጀግና የኢትዮጵያ አርበኛ ነበር።"


# Initialize all processors
cleaner = AmharicCleaner()
normalizer = AmharicNormalizer()
lemmatizer = AmharicLemmatizer()
stemmer = AmharicStemmer()
stopword_processor = AmharicStopwordProcessor()

# Apply each processing step
print("Original Text:", sample_text)

# 1. Clean the text
cleaned_texth = cleaner.remove_html(sample_text)

print("\nAfter HTML Removal:", cleaned_texth)
cleaned_textn = cleaner.remove_noise(cleaned_texth)
print("\nAfter Cleaning:", cleaned_textn)




# 2. Normalize the text
normalized_text = normalizer.normalize_amharic_chars(cleaned_textn)
normalized_text2=normalizer.normalize_punctuation_spacing(normalized_text)
                                                
normalized_text3=normalizer.expand_abbreviations(normalized_text2)
print("\nAfter Normalization:", normalized_text3)



# 4. Stopword removal
without_stopwords = stopword_processor.remove_stopwords(normalized_text3)
print("\nAfter Stopword Removal:", without_stopwords)

stem= stemmer.stem_amharic(without_stopwords)
print("\nAfter Stemming:", stem)
# 5. Lemmatization
#lemmatized_text = [lemmatizer.lemmatize(word) for word in without_stopwords.split()]
#print("\nAfter Lemmatization:", " ".join(lemmatized_text))

# 6. Stemming
##print("\nAfter Stemming:", " ".join(stemmed_text))