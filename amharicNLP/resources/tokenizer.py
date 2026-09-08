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


    # NOTE: an earlier version of this tokenizer force-split every word
    # starting with a preposition-like prefix (የ, በ, ከ, ለ, ...) into two
    # tokens, e.g. "የካቲት" -> "የ ካቲት" or "በለስ" -> "በ ለስ". That is wrong for
    # the (large) set of Amharic words that only coincidentally start with
    # those letters, and it can't be done correctly without a lexicon.
    # Splitting a bound prefix off a word is morphological analysis, not
    # tokenization -- that's handled by AmharicStemmerprocess in
    # stemmer.py, which strips prefixes/suffixes with a protected-word
    # list and a minimum stem length instead of a blind regex.

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

        # Split on whitespace and return tokens
        tokens = text.strip().split()
        return tokens


