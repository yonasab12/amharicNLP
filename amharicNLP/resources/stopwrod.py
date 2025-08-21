


from .amharic_stopword import data as stop_words
class  AmharicStopwordProcessor:
    def __init__(self):
        self.stop_words = set(stop_words)

    def remove(self, text):
        tokens = text.split()
        return [token for token in tokens if token not in self.stop_words]
