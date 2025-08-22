import os
from . import data_lemma

class AmharicLemmatizer:
    def __init__(self, wordnet_data=None):
        if wordnet_data is None:
            # Access the data directly from the imported module
            self.wordnet_data = data_lemma.data
        else:
            self.wordnet_data = wordnet_data

    def lemmatize(self, word):
        for entry in self.wordnet_data:
            if word in entry["example"]:
                return entry["lemma"]
        return word