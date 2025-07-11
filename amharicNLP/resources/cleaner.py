from.utils import AmharicLanguageDetector
import re

class AmharicCleaner:
    """import re
from utils import is_amharic_char
    A class for cleaning text by removing HTML tags, emojis, and non-Amharic characters.
    """
    
    def __init__(self):
        # Compile regex patterns once when the class is instantiated
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  
            u"\U0001F300-\U0001F5FF"  
            u"\U0001F680-\U0001F6FF"  
            u"\U0001F1E0-\U0001F1FF"  
            u"\U00002500-\U00002BEF"  
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  
            u"\u3030"
            "]+", flags=re.UNICODE)
        
        self.html_pattern = re.compile(r'<.*?>')

    def remove_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        return self.html_pattern.sub('', text)

    def remove_noise(self, text: str, keep_amharic: bool = True) -> str:
        """
        Remove non-Amharic characters, emojis, and special symbols while preserving Amharic abbreviations.
        
        Args:
            text: Input text to clean
            keep_amharic: If True, keeps only Amharic characters, numbers, basic punctuation and abbreviations
        """
        text = self.emoji_pattern.sub(r'', text)
        
        if keep_amharic:
            text = ''.join(char for char in text if (
                AmharicLanguageDetector.is_amharic_char(char) or 
                char.isdigit() or 
                char in '.,!?;:- ' or
                char in '/.'  
            ))
        
        text = ' '.join(text.split())
        return text