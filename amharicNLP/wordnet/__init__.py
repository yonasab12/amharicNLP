
from ..resources.lemmatizer import AmharicLemmatizer  
from data_lemma import data

lemmatizer = AmharicLemmatizer()

# Make available at package level
__all__ = ['data', 'lemmatizer']