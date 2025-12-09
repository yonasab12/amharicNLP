import re

class AmharicWordTokenizer:
    """
    Improved Tokenizer for Amharic text.
    Handles punctuation, opening/closing quotes, and parentheses/brackets.
    """

    # Opening quotes anywhere attached
    OPENING_QUOTES = [
        (re.compile(r'([‹“"`«‛‘])(?!\s|[‹“"`«‛‘])'), r'\1 '),  # space after the quote
    ]

    # Closing quotes anywhere attached
    CLOSING_QUOTES = [
        (re.compile(r'([^\s])([›”"»’])'), r'\1 \2'),  # space before the quote
    ]
    # Amharic punctuation marks
    PUNCTUATION = [
        (re.compile(r"\'"), r" ' "),
        (re.compile(r'"'), r' " '),
        (re.compile(r'፡'), r' ፡ '),
        (re.compile(r'[!?]'), r' \g<0> '),
        (re.compile(r'።'), r' ። '),
        (re.compile(r'፣'), r' ፣ '),
        (re.compile(r'፤'), r' ፤ '),
        (re.compile(r'≈'), r' ≈ '),
        (re.compile(r'$'), r' $ '),
        (re.compile(r'[-–—]+'), r' \g<0> '),
    ]

    # Parentheses and brackets
    PARENS_BRACKETS = [
        (re.compile(r'([\(\)\[\]\{\}])'), r' \1 ')
    ]

    def tokenize(self, text):
        # Separate opening quotes
        for regexp, substitution in self.OPENING_QUOTES:
            text = regexp.sub(substitution, text)

        # Separate closing quotes
        for regexp, substitution in self.CLOSING_QUOTES:
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
