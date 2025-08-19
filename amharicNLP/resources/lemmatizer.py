import os
import json
import amharicNLP
print(amharicNLP.__file__)
class AmharicLemmatizer:
    def __init__(self, wordnet_data=None):
        if wordnet_data is None:
            # Load default WordNet data from the package
            wordnet_path = os.path.join(os.path.dirname(__file__), 'data_lemma.py')
            with open(wordnet_path, encoding='utf-8') as f:
                self.wordnet_data = json.load(f)
        else:
            self.wordnet_data = wordnet_data

    def lemmatize(self, word):
        for entry in self.wordnet_data:
            if word in entry["example"]:
                return entry["lemma"]
        return word