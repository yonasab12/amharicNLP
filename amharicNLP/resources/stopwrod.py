import os
import re

class AmharicStopwordProcessor:
    def __init__(self, stopwords_file=None):
        """
        Initialize stopword processor with optional custom stopwords file
        :param stopwords_file: Optional path to custom stopwords file
        """
        if stopwords_file is None:
            # Automatically locate the stopwords file in package resources
            current_dir = os.path.dirname(os.path.abspath(__file__))
            stopwords_file = os.path.join(current_dir, "amharic_stopwords.txt")
        
        self.stopwords = self.load_stopwords(stopwords_file)
    
    @staticmethod
    def load_stopwords(file_path):
        """Load stopwords from a text file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Stopwords file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            return set(line.strip() for line in file)
    
    def remove_stopwords(self, text):
        """
        Remove Amharic stopwords from text
        :param text: Input text to process
        :return: Filtered text with stopwords removed
        """
        # Tokenize while preserving Amharic characters
        words = re.findall(r'[\w\u1200-\u137F]+', text, re.UNICODE)
        
        # Filter out stopwords (case-insensitive)
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        
        return ' '.join(filtered_words)