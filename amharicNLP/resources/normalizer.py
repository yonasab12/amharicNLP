import re
import unicodedata
from typing import Dict, Optional, List, Tuple, Set

class AmharicNormalizer:
   
    AMHARIC_NORMALIZATION_MAP = {
        # Normalize variants of ሀ series
        'ሃ': 'ሀ', 'ኅ': 'ሀ', 'ኃ': 'ሀ', 'ሐ': 'ሀ', 'ሓ': 'ሀ', 'ኻ': 'ሀ',
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
        'ፍ/ስላሴ': 'ፍቅረ �ስላሴ',
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

    # ====================== CORE FUNCTIONS ======================

    @staticmethod
    def normalize_unicode(text: str, form: str = 'NFC') -> str:
        """Normalize Unicode text to specified form (default: NFC)."""
        return unicodedata.normalize(form, text)

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
        """
        if not to_western:
            return text
        
        return ''.join(self.AMHARIC_NUMERALS.get(char, char) for char in text)

    def fix_common_spelling(self, text: str, custom_map: Optional[Dict[str, str]] = None) -> str:
        """
        Fix common Amharic spelling variations.
        
        Args:
            custom_map: Additional spelling corrections to apply
        """
        corrections = self.COMMON_SPELLING_CORRECTIONS.copy()
        if custom_map:
            corrections.update(custom_map)
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
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
        custom_abbr_map: Optional[Dict[str, str]] = None
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
        
        Returns:
            Normalized text string
        """
        # Normalize Unicode first
        text = self.normalize_unicode(text, unicode_form)
        
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