import re
import unicodedata
from typing import Dict, Optional, List, Tuple, Set

class AmharicNormalizer:
   
    AMHARIC_NORMALIZATION_MAP = {
        # Normalize variants of ሀ series.
        # NOTE: 'ኅ' (the 6th-order/bare form of the ኀ series) previously
        # appeared twice in this dict, once mapped to 'ሀ' and once to 'ህ' --
        # a duplicate dict key, so Python silently kept only the second
        # ('ህ') and the first definition was dead. 'ህ' is correct here:
        # 'ኅ' is the 6th-order form, matching 'ሕ' -> 'ህ' below (also
        # 6th-order), not the base/1st-order 'ሀ'.
        'ሃ': 'ሀ', 'ኃ': 'ሀ', 'ሐ': 'ሀ', 'ሓ': 'ሀ', 'ኻ': 'ሀ',
        'ሑ': 'ሁ', 'ኁ': 'ሁ', 'ዅ': 'ሁ',
        'ኂ': 'ሂ', 'ሒ': 'ሂ', 'ኺ': 'ሂ',
        'ኌ': 'ሄ', 'ሔ': 'ሄ', 'ዄ': 'ሄ',
        'ሕ': 'ህ', 'ኅ': 'ህ',
        'ኆ': 'ሆ', 'ሖ': 'ሆ', 'ኾ': 'ሆ',
        
        # Normalize ሰ series
        'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 
        'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ',
        
        # Normalize አ series
        'ዓ': 'አ', 'ኣ': 'አ', 'ዐ': 'አ',
        'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ',
        
        # Normalize ፀ series
        'ጸ': 'ፀ', 'ጹ': 'ፁ', 'ጺ': 'ፂ', 'ጻ': 'ፃ',
        'ጼ': 'ፄ', 'ጽ': 'ፅ', 'ጾ': 'ፆ',
        
        # Other normalizations
        'ቊ': 'ቁ', 'ኵ': 'ኩ'
    }

    # Labialized character components
    LABIALIZED_COMPONENTS = {
        'ሉ': 'ሏ', 'ሙ': 'ሟ', 'ቱ': 'ቷ', 'ሩ': 'ሯ', 'ሱ': 'ሷ', 'ሹ': 'ሿ',
        'ቁ': 'ቋ', 'ቡ': 'ቧ', 'ቹ': 'ቿ', 'ሁ': 'ኋ', 'ኑ': 'ኗ', 'ኙ': 'ኟ',
        'ኩ': 'ኳ', 'ዙ': 'ዟ', 'ጉ': 'ጓ', 'ደ': 'ዷ', 'ጡ': 'ጧ', 'ጩ': 'ጯ',
        'ጹ': 'ጿ', 'ፉ': 'ፏ'
    }

    # Punctuation Handling
    AMHARIC_PUNCTUATION = {'።', '፣', '፤', '፥', '፦', '፡'}  # Amharic-specific punctuation

    PUNCTUATION_NORMALIZATION = [
        # "፡፡" (two Ethiopic wordspace marks together) is a traditional way
        # of writing the sentence-final full stop, still common in news
        # writing -- equivalent in meaning to "።". This MUST run before the
        # single-"፡" rule below, or "፡፡" gets split into two separate
        # spaced tokens ("፡ ፡") instead of being recognized as one full
        # stop. Found by testing against real transcripts from the
        # snapwre/amharic-speech dataset, where it appeared in roughly a
        # quarter of sampled sentences.
        (re.compile(r'፡፡'), '።'),
        (re.compile(r'\.'), ' . '), 
        (re.compile(r'፡'), ' ፡ '),
        (re.compile(r'[!?]'), r' \g<0> '),  
        (re.compile(r'።'), ' ። '),
        (re.compile(r'፣'), ' ፣ '),
        (re.compile(r'፤'), ' ፤ '),
        (re.compile(r'[-–—]+'), r' \g<0> '),
    ]

    # Numerals
    AMHARIC_NUMERALS = {
        '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
        '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10',
        '፳': '20', '፴': '30', '፵': '40', '፶': '50',
        '፷': '60', '፸': '70', '፹': '80', '፺': '90', '፻': '100'
    }

    # Spelling Corrections
    COMMON_SPELLING_CORRECTIONS = {
        'አበ': 'አበራ',
        'ሠላም': 'ሰላም',
        # Add more common corrections here
    }

    # Abbreviations
    AMHARIC_ABBREVIATIONS = {
        'ት/ቤት': 'ትምህርት ቤት',
        'ት/ርት': 'ትምህርት',
        'ት/ክፍል': 'ትምህርት ክፍል',
        'ሃ/አለቃ': 'ሀምሳ አለቃ',
        'ሃ/ስላሴ': 'ሀይለ ስላሴ',
        'ደ/ዘይት': 'ደብረ ዘይት',
        'ደ/ታቦር': 'ደብረ ታቦር',
        'መ/ር': 'መምህር',
        'መ/ቤት': 'መስሪያ ቤት',
        'መ/አለቃ': 'መቶ አለቃ',
        'ክ/ከተማ': 'ክፍለ ከተማ',
        'ክ/ሀገር': 'ክፍለ ሀገር',
        'ወ/ር': 'ወታደር',
        'ወ/ሮ': 'ወይዘሮ',
        'ወ/ሪት': 'ወይዘሪት',
        'ወ/ስላሴ': 'ወሌተ ስላሴ',
        'ፍ/ስላሴ': 'ፍቅረ ስላሴ',
        'ፍ/ቤት': 'ፍርድ ቤት',
        'ጽ/ቤት': 'ጽህፈት ቤት',
        'ሲ/ር': 'ሲስተር',
        'ጠ/ሚኒስትር': 'ጠቅላይ ሚኒስትር',
        'ዶ/ር': 'ዶክተር',
        'ገ/ጊዮርጊስ': 'ገብረ ጊዮርጊስ',
        'ቤ/ክርስትያን': 'ቤተ ክርስትያን',
        'ም/ስራ': 'ምክትል ስራ',
        'ም/ቤት': 'ምክር ቤት',
        'ተ/ሃይማኖት': 'ተክለ ሃይማኖት',
        'ሚ/ር': 'ሚኒስትር',
        'ኮ/ል': 'ኮለኔል',
        'ሜ/ጀነራል': 'ሜጀር ጀነራል',
        'ብ/ጀነራል': 'ብርጋዳር ጀነራል',
        'ሌ/ኮለኔል': 'ሌተናንት ኮለኔል',
        'ሊ/መንበር': 'ሊቀ መንበር',
        'አ/አ': 'አዲስ አበባ',
        'ር/መምህር': 'ርእሰ መምህር',
        'ፕ/ት': 'ፕሬዝዳንት',
        'ዓ.ም': 'አመተ ምህረት',
        'ዓ.ዓ': 'አዲስ አበባ',
        'ዶ.ር': 'ዶክተር',
        'ፕ/ር': 'ፕሮፌሰር'
    }

    # Social-media / noisy-text patterns. Compiled once at class level
    # since they don't depend on instance state.
    URL_RE = re.compile(
        r'\b(?:https?://|www\.)[^\s<>"\']+', re.IGNORECASE
    )
    EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
    HASHTAG_RE = re.compile(r'(?<!\w)#([\w\u1200-\u137F]+)', re.UNICODE)
    MENTION_RE = re.compile(r'(?<!\w)@([\w\u1200-\u137F]+)', re.UNICODE)

    # Repeated Amharic/Latin letters ("ደሓናናናና" -> "ደሓና", "sooooo" ->
    # "soo"): collapse runs of 3+ identical letters down to 2, which
    # keeps intentional doubles (Amharic geminated-sounding spellings,
    # English "soo") while cleaning up typing noise/emphasis stretching.
    _REPEATED_CHAR_RE = re.compile(r'(\w)\1{2,}', re.UNICODE)
    # Repeated punctuation ("really???","wait...","አይ!!!") collapsed to a
    # single instance of that punctuation mark.
    _REPEATED_PUNCT_RE = re.compile(r'([!?.,;:፣፤፥፦])\1+')

    # ====================== CORE FUNCTIONS ======================

    @staticmethod
    def normalize_unicode(text: str, form: str = 'NFC') -> str:
        """Normalize Unicode text to specified form (default: NFC)."""
        return unicodedata.normalize(form, text)

    def normalize(self, text: str) -> str:
        """Alias for normalize_amharic_chars (character-level normalization
        only, e.g. ኣ -> አ). Kept as the short, discoverable entry point."""
        return self.normalize_amharic_chars(text)

    def normalize_amharic_chars(self, text: str) -> str:
        """Normalize Amharic characters to their canonical forms with enhanced labialization handling."""
        normalized = []
        i = 0
        n = len(text)
        
        while i < n:
            current_char = text[i]
            
            # Check for labialized character patterns (base + ዋ or አ)
            if i + 1 < n and current_char in self.LABIALIZED_COMPONENTS:
                next_char = text[i+1]
                if next_char in ['ዋ', 'አ']:
                    normalized.append(self.LABIALIZED_COMPONENTS[current_char])
                    i += 2
                    continue
            
            # Handle standard character mappings
            normalized.append(self.AMHARIC_NORMALIZATION_MAP.get(current_char, current_char))
            i += 1
        
        return ''.join(normalized)

    def normalize_punctuation_spacing(self, text: str) -> str:
        """Ensure proper spacing around punctuation marks."""
        for pattern, replacement in self.PUNCTUATION_NORMALIZATION:
            text = pattern.sub(replacement, text)
        return text

    def remove_punctuation(self, text: str, 
                         keep_basic: bool = True,
                         keep_amharic: bool = True) -> str:
        """
        Enhanced punctuation removal with Amharic support.
        
        Args:
            text: Input text
            keep_basic: Keep basic punctuation (.!?,;:-)
            keep_amharic: Keep Amharic-specific punctuation
        """
        keep = []
        if keep_basic:
            keep.append(r'\.!?,;:-')
        if keep_amharic:
            keep.append(''.join(self.AMHARIC_PUNCTUATION))
        
        if keep:
            punctuation = rf'[^\w\s{"".join(keep)}]'
        else:
            punctuation = r'[^\w\s]'
        
        return re.sub(punctuation, '', text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize all whitespace characters to single space."""
        text = re.sub(r'[\s\u2000-\u200F\u2028-\u202F\u205F\u3000]+', ' ', text)
        return text.strip()

    def normalize_numbers(self, text: str, to_western: bool = False) -> str:
        """
        Normalize Amharic numbers to either Western or keep Amharic.

        Args:
            to_western: If True, converts Amharic numbers to Western (0-9)

        LIMITATION: this does a per-character substitution
        (AMHARIC_NUMERALS is a 1:1 symbol map), which is only correct for
        single-symbol numbers (፩-፱, ፲, ፳...፻). The Ge'ez numeral system is
        additive/multiplicative, not positional like Arabic numerals, so a
        composite number like ፲፱ ("nineteen" = 10 + 9) would be converted
        symbol-by-symbol into "109" (concatenating "10" and "9") instead of
        the correct "19". Do not rely on this for text containing multi-
        symbol Ge'ez numbers without verifying the output; a correct
        converter needs to parse the additive/multiplicative grouping, not
        just substitute characters.
        """
        if not to_western:
            return text
        
        return ''.join(self.AMHARIC_NUMERALS.get(char, char) for char in text)

    def normalize_urls(self, text: str, mode: str = 'placeholder', placeholder: str = '<URL>') -> str:
        """
        Handle URLs in text.

        Args:
            mode: 'placeholder' replaces each URL with `placeholder`
                  (default -- keeps a slot for the fact a URL was there,
                  without polluting a tokenizer/vocab with raw links);
                  'remove' deletes URLs outright; 'keep' leaves text
                  unchanged (useful when you only want the other
                  normalize_* toggles).
        """
        if mode == 'keep':
            return text
        if mode not in ('placeholder', 'remove'):
            raise ValueError("mode must be 'placeholder', 'remove', or 'keep'")
        replacement = placeholder if mode == 'placeholder' else ''
        return self.URL_RE.sub(replacement, text)

    def normalize_emails(self, text: str, mode: str = 'placeholder', placeholder: str = '<EMAIL>') -> str:
        """Handle email addresses in text. Same `mode` semantics as normalize_urls."""
        if mode == 'keep':
            return text
        if mode not in ('placeholder', 'remove'):
            raise ValueError("mode must be 'placeholder', 'remove', or 'keep'")
        replacement = placeholder if mode == 'placeholder' else ''
        return self.EMAIL_RE.sub(replacement, text)

    def normalize_hashtags(self, text: str, mode: str = 'strip_symbol') -> str:
        """
        Handle hashtags (#ኢትዮጵያ, #Ethiopia) in text.

        Args:
            mode: 'strip_symbol' keeps the tag word, drops the '#' (the
                  word is often meaningful running text, e.g. a slogan or
                  topic, so dropping it loses information most Amharic
                  social-media pipelines want to keep); 'remove' deletes
                  the whole hashtag including the word; 'keep' leaves
                  text unchanged.
        """
        if mode == 'keep':
            return text
        if mode == 'strip_symbol':
            return self.HASHTAG_RE.sub(r'\1', text)
        if mode == 'remove':
            return self.HASHTAG_RE.sub('', text)
        raise ValueError("mode must be 'strip_symbol', 'remove', or 'keep'")

    def normalize_mentions(self, text: str, mode: str = 'placeholder', placeholder: str = '<USER>') -> str:
        """
        Handle @mentions in text.

        Args:
            mode: 'placeholder' replaces each mention with `placeholder`
                  (default -- a mentioned username is usually not
                  linguistically meaningful content, unlike a hashtag
                  word, and often a real name that shouldn't be included
                  verbatim in a training corpus); 'remove' deletes the
                  mention outright; 'strip_symbol' keeps the username
                  text, drops the '@'; 'keep' leaves text unchanged.
        """
        if mode == 'keep':
            return text
        if mode == 'strip_symbol':
            return self.MENTION_RE.sub(r'\1', text)
        if mode == 'remove':
            return self.MENTION_RE.sub('', text)
        if mode == 'placeholder':
            return self.MENTION_RE.sub(placeholder, text)
        raise ValueError("mode must be 'placeholder', 'strip_symbol', 'remove', or 'keep'")

    def collapse_repeated_characters(self, text: str, max_repeat: int = 2) -> str:
        """
        Collapse runs of 3+ identical repeated characters down to
        `max_repeat` (default 2), e.g. "ደሓናናናና" -> "ደሓናና",
        "sooooo good" -> "soo good". Guards against typing-noise/emphasis
        stretching inflating the vocabulary with near-duplicate word
        forms, while still allowing an intentional double letter through.
        """
        if max_repeat < 1:
            raise ValueError("max_repeat must be >= 1")

        def _collapse(match: "re.Match") -> str:
            return match.group(1) * max_repeat

        return self._REPEATED_CHAR_RE.sub(_collapse, text)

    def collapse_repeated_punctuation(self, text: str) -> str:
        """Collapse runs of repeated punctuation to a single mark, e.g.
        "እውነት ነው???" -> "እውነት ነው?", "ጠብቅ...." -> "ጠብቅ.", handling both
        Latin (!?.,;:) and Amharic (፣ ፤ ፥ ፦) punctuation."""
        return self._REPEATED_PUNCT_RE.sub(r'\1', text)

    def fix_common_spelling(self, text: str, custom_map: Optional[Dict[str, str]] = None) -> str:
        """
        Fix common Amharic spelling variations.
        
        Args:
            custom_map: Additional spelling corrections to apply
        """
        corrections = self.COMMON_SPELLING_CORRECTIONS.copy()
        if custom_map:
            corrections.update(custom_map)

        # Word-boundary-aware replacement, not a blind substring replace.
        # str.replace() would also match 'wrong' inside unrelated longer
        # words (e.g. correcting 'አበ' -> 'አበራ' would also mangle 'አበባ'
        # "flower" into 'አበራባ', since 'አበ' is a substring of it). \b is
        # Unicode-aware in Python's re module by default, so it works
        # correctly against Ethiopic script boundaries (whitespace/
        # punctuation), not just ASCII word chars.
        for wrong, correct in sorted(corrections.items(), key=lambda kv: len(kv[0]), reverse=True):
            text = re.sub(rf'\b{re.escape(wrong)}\b', correct, text)

        return text

    def expand_abbreviations(self, text: str, custom_abbr: Optional[Dict[str, str]] = None) -> str:
        """
        Expand Amharic abbreviations to their full forms.
        
        Args:
            text: Input text containing abbreviations
            custom_abbr: Additional abbreviations to expand
        
        Returns:
            Text with abbreviations expanded
        """
        abbreviations = self.AMHARIC_ABBREVIATIONS.copy()
        if custom_abbr:
            abbreviations.update(custom_abbr)
        
        # Sort by length to match longer abbreviations first
        for abbr in sorted(abbreviations, key=len, reverse=True):
            text = text.replace(abbr, abbreviations[abbr])
        
        return text

    # ====================== MAIN NORMALIZATION FUNCTION ======================

    def normalize_text(
        self,
        text: str,
        *,
        unicode_form: str = 'NFC',
        normalize_chars: bool = True,
        normalize_punct_spacing: bool = True,
        remove_punct: bool = False,
        keep_basic_punct: bool = True,
        keep_amharic_punct: bool = True,
        normalize_ws: bool = True,
        normalize_nums: bool = False,
        fix_spelling: bool = False,
        expand_abbr: bool = True,
        custom_spelling_map: Optional[Dict[str, str]] = None,
        custom_abbr_map: Optional[Dict[str, str]] = None,
        url_mode: Optional[str] = None,
        email_mode: Optional[str] = None,
        hashtag_mode: Optional[str] = None,
        mention_mode: Optional[str] = None,
        collapse_repeated_chars: bool = False,
        max_repeat: int = 2,
        collapse_repeated_punct: bool = False,
    ) -> str:
        """
        Comprehensive Amharic text normalization pipeline.
        
        Args:
            text: Input text to normalize
            unicode_form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
            normalize_chars: Normalize Amharic character variants
            normalize_punct_spacing: Add spaces around punctuation
            remove_punct: Remove punctuation
            keep_basic_punct: Keep basic punctuation if removing punctuation
            keep_amharic_punct: Keep Amharic punctuation if removing punctuation
            normalize_ws: Normalize whitespace
            normalize_nums: Convert Amharic numbers to Western numerals
            fix_spelling: Apply common spelling corrections
            expand_abbr: Expand common Amharic abbreviations
            custom_spelling_map: Custom spelling corrections dictionary
            custom_abbr_map: Custom abbreviations dictionary
            url_mode: If set ('placeholder'/'remove'/'keep'), run
                normalize_urls with this mode. None (default) skips URL
                handling entirely -- opt-in, since it's social-media/noisy
                -text specific and not every corpus has URLs.
            email_mode: Same as url_mode, for normalize_emails.
            hashtag_mode: If set ('strip_symbol'/'remove'/'keep'), run
                normalize_hashtags with this mode. None skips it.
            mention_mode: If set ('placeholder'/'strip_symbol'/'remove'/
                'keep'), run normalize_mentions with this mode. None skips it.
            collapse_repeated_chars: Collapse runs of 3+ repeated letters
                (typing noise/emphasis stretching) down to `max_repeat`.
            max_repeat: Max repeats kept when collapse_repeated_chars is True.
            collapse_repeated_punct: Collapse runs of repeated punctuation
                (e.g. "???", "....") to a single mark.
        
        Returns:
            Normalized text string
        """
        # Normalize Unicode first
        text = self.normalize_unicode(text, unicode_form)

        # Social-media / noisy-text handling runs early, before
        # abbreviation expansion and punctuation spacing, so URLs/emails
        # aren't mangled by the '.'-spacing rule below and hashtags/
        # mentions are resolved to plain words before tokniz-adjacent
        # steps see them.
        if url_mode is not None:
            text = self.normalize_urls(text, mode=url_mode)
        if email_mode is not None:
            text = self.normalize_emails(text, mode=email_mode)
        if hashtag_mode is not None:
            text = self.normalize_hashtags(text, mode=hashtag_mode)
        if mention_mode is not None:
            text = self.normalize_mentions(text, mode=mention_mode)

        if collapse_repeated_punct:
            text = self.collapse_repeated_punctuation(text)
        if collapse_repeated_chars:
            text = self.collapse_repeated_characters(text, max_repeat=max_repeat)

        # Expand abbreviations
        if expand_abbr:
            text = self.expand_abbreviations(text, custom_abbr_map)
        
        # Normalize Amharic characters
        if normalize_chars:
            text = self.normalize_amharic_chars(text)
        
        # Normalize punctuation spacing
        if normalize_punct_spacing:
            text = self.normalize_punctuation_spacing(text)
        
        # Remove punctuation (if enabled)
        if remove_punct:
            text = self.remove_punctuation(text, 
                                        keep_basic=keep_basic_punct,
                                        keep_amharic=keep_amharic_punct)
        
        # Normalize whitespace
        if normalize_ws:
            text = self.normalize_whitespace(text)
        
        # Normalize numbers
        if normalize_nums:
            text = self.normalize_numbers(text, to_western=True)
        
        # Fix common spelling
        if fix_spelling:
            text = self.fix_common_spelling(text, custom_spelling_map)
        
        return text