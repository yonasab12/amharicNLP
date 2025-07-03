class AmharicLemmatizer:
    def __init__(self, wordnet_data):
        self.synsets = wordnet_data
    
    def lemmatize(self, word, pos=None):
        """Simple dictionary-based lemmatization"""
        for syn in self.synsets:
            if word in syn["lemmas"]:
                if pos is None or syn["pos"] == pos:
                    return syn["lemmas"][0]
        return word  # Return original if not found