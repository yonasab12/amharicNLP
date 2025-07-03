import re

class AmharicWordTokenizer:
    """
    Tokenizer for Amharic text.
    Handles Amharic punctuation, contractions, and parentheses/brackets.
    """
    
    # Amharic starting quotes and punctuation
    STARTING_QUOTES = [
        (re.compile(r'^«'), r'`'),
        (re.compile(r'^»'), r"'"),
        (re.compile(r'^"'), r''),
    ]

    # Amharic punctuation marks

 
    PUNCTUATION= [ 
        (re.compile(r'\.'), r' . '), 
        (re.compile(r'፡'), r' ፡ '),
        (re.compile(r'[!?]'), r' \g<0> '),  
        (re.compile(r'።'), r' ። '),
        (re.compile(r'፣'), r' ፣ '),
        (re.compile(r'፤'), r' ፤ '),
        (re.compile(r'[-–—]+'), r' \g<0> '),
    ]

    # Parentheses and brackets (replace with labels)
  
                        
    PARENS_BRACKETS = [
       (re.compile(r'([\(\)\[\]\{\}])'), r' \1 ')
    ]


    # Amharic contractions and compound words
    CONTRACTIONS = [
        (re.compile(r'(?i)\b(የ|በ|ከ|ለ|ውስጥ|ላይ|በላይ)([^\s]+)\b'), r'\1 \2'),  # Prepositions
        (re.compile(r'(?i)\b(እኔ|አንቺ|እሱ|እሷ)([^\s]+)\b'), r'\1 \2'),  # Pronouns
    ]

    def tokenize(self, text):
        # Apply starting quotes
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        # Handle punctuation
        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Replace parentheses/brackets
        for regexp, substitution in self.PARENS_BRACKETS:
            text = regexp.sub(substitution, text)

        # Split contractions and compound words
        for regexp, substitution in self.CONTRACTIONS:
            text = regexp.sub(substitution, text)

        # Split on whitespace and return tokens
        tokens = text.strip().split()
        return tokens


