import re
import unicodedata

from .utils import AmharicLanguageDetector


class AmharicCleaner:
    """A class for cleaning text by removing HTML tags, emojis, and
    non-Amharic characters."""

    def __init__(self):
        # Compile regex patterns once when the class is instantiated.
        #
        # NOTE: this is deliberately restricted to actual emoji/pictograph
        # blocks (faces, symbols, transport, flags, dingbats, supplemental
        # symbols). An earlier version of this pattern also included the
        # blanket range \U00010000-\U0010ffff, which is not an emoji range
        # at all -- it covers essentially every astral-plane block (CJK
        # extensions, historic scripts, musical/mathematical symbols, ...)
        # and would have silently stripped any of that if remove_noise()
        # were ever called with keep_amharic=False. Narrowed to the actual
        # emoji ranges below; the emoji_pattern's job is "remove emoji", not
        # "remove everything above the BMP".
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"   # emoticons
            u"\U0001F300-\U0001F5FF"   # symbols & pictographs
            u"\U0001F680-\U0001F6FF"   # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"   # flags (iOS)
            u"\U0001F900-\U0001F9FF"   # supplemental symbols & pictographs
            u"\U0001FA70-\U0001FAFF"   # symbols & pictographs extended-A
            u"\U00002500-\U00002BEF"   # box drawing / misc symbols & arrows
            u"\U00002702-\U000027B0"   # dingbats
            u"\U000024C2-\U0001F251"
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

    def remove_diacritics(self, text: str) -> str:
        """
        Strip Unicode combining diacritical marks (accents, umlauts, etc.)
        from text via NFKD decomposition + combining-mark removal.

        NOTE: this has essentially no effect on Amharic/Ge'ez script -- each
        Ethiopic syllable (fidel) is its own precomposed codepoint, there
        are no separate combining vowel marks to strip, so Amharic text is
        returned unchanged. This is useful for the Latin-script portions of
        mixed-language text (e.g. normalizing "café" -> "cafe" in a
        code-switched sentence), not for Amharic text itself.
        """
        decomposed = unicodedata.normalize('NFKD', text)
        return ''.join(ch for ch in decomposed if not unicodedata.combining(ch))
