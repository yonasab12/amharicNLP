import re
class AmharicStopwordProcessor:
    def load_stopwords(file_path):
        """Load stopwords from a text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            stopwords = set(line.strip() for line in file)
        return stopwords

    def remove_amharic_stopwords(text, stopwords_file=r'C:\Users\LENOVO\Desktop\webproject\amharicNLP\amharicNLP\resources\amharic_stopwords.txt'):
        """
        Remove Amharic stopwords from text.
        
        Args:
            text (str): Input text to process.
            stopwords_file (str): Path to the stopwords .txt file.
        
        Returns:
            str: Filtered text with stopwords removed.
        """
      
        try:
            amharic_stopwords = AmharicStopwordProcessor.load_stopwords(stopwords_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Stopwords file not found: {stopwords_file}")
        
      
        words = re.findall(r'[\w-]+', text, re.UNICODE)
        

        filtered_words = [word for word in words if word.lower() not in amharic_stopwords]
        
 
        filtered_text = ' '.join(filtered_words)
        
        return filtered_text
