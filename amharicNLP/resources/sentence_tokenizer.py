import re

class AmharicSentenceTokenizer:
    """
    Sentence tokenizer for Amharic text.
    Handles:
      - Sentence-ending punctuation (።, !, ?, :, ፡)
      - Combo endings (?!, !?, !!, ??)
      - Interjections (e.g., እሺ!, አዎ!)
      - ? used as uncertainty marker (inside parentheses or after /)
    """

    def __init__(self):
        # Interjections often appearing with !
        self.interjections = [
            "እሺ", "አዎ", "ወዴ", "ውይ", "እንዴ", "እንግዲኛ", "እሷ", "እሱ", "እኔ"
        ]

        # Combo endings (?!, !?, !!, ??)
        self.combo_end = re.compile(r'([!?]\?|\?!|!!|\?\?)')

        # Question mark safe (do not split if followed by ) or /)
        self.question_safe = re.compile(r'\?(?![\)/])')

        # Other sentence-ending punctuation
        self.sentence_end_pattern = re.compile(r'(።|!|:{2,}|፡{2,})')

        # Interjection protection
        self.interj_pattern = re.compile(
            r'\b(' + '|'.join(map(re.escape, self.interjections)) + r')!'
        )

    def split_sentences(self, text):
        # Protect combo endings
        text = self.combo_end.sub(lambda m: f"@@COMBO@@{m.group(0)}", text)

        # Protect interjections
        text = self.interj_pattern.sub(lambda m: f"@@INTERJ@@{m.group(0)}", text)

        # Handle ? carefully
        text = self.question_safe.sub(' ? ', text)

        # Split on other sentence-ending punctuation
        text = self.sentence_end_pattern.sub(lambda m: f' {m.group(0)} ', text)

        # Restore combo endings
        text = text.replace('@@COMBO@@', '')

        # Restore interjections
        text = text.replace('@@INTERJ@@', '')

        # Split on whitespace for sentence boundaries
        parts = re.split(r'\s+', text)

        # Reconstruct sentences based on punctuation
        sentences = []
        current = []
        for part in parts:
            if not part:
                continue
            current.append(part)
            if re.match(r'(።|\?|!|:{2,}|፡{2,})$', part):
                sentences.append(' '.join(current).strip())
                current = []
        if current:
            sentences.append(' '.join(current).strip())
        return sentences
