class AmharicLanguageDetector:
    def is_amharic_char(char: str) -> bool:
        """Check if a character is in the Amharic Unicode block."""
        # Amharic Unicode range: U+1200 to U+137F
        return '\u1200' <= char <= '\u137F'

    def is_amharic_text(text: str, threshold: float = 0.7) -> bool:
        """
        Check if text is primarily Amharic.
        
        Args:
            text: Input text to check
            threshold: Minimum proportion of Amharic characters to consider as Amharic text
        """
        if not text:
            return False
        
        amharic_count = sum(1 for char in text if AmharicLanguageDetector.is_amharic_char(char))
        total_chars = len(text)
        
        return (amharic_count / total_chars) >= threshold