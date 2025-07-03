class AmharicStemmer:
    # Comprehensive Amharic Stemmer
    # Based on linguistic research from Addis Ababa University and Amharic NLP Toolkit

    # Curated affix lists from academic sources
    PREFIXES = [
        # Subject markers
        "እን", "ይ", "ት", "እ", "ም", "ተ", "አል", "አት", "አይ", "ን", 
        "ታ", "ይተ", "አል", "አም", "ተም", "አ",
        
        # Prepositions and conjunctions
        "የ", "በ", "ከ", "ለ", "ስለ", "እስከ", "በስተ", "በኩል", "በውስጥ", 
        "ከውስጥ", "ለምሳሌ", "እንደ", "ያለ", "ከላይ", "በላይ",
        
        # Verb derivation markers
        "አስ", "ተስ", "አስተ", "ማ", "መ", "ታ", "ት", "ብ", "ን", "ድ", 
        "ግ", "ል", "ም", "ስ", "ቅ", "ት", "ው", "ዝ", "ሻ", "ካ", "ያ", 
        "ዋ", "ኃ", "ኅ", "ኻ", "ል", "ት", "ክ"
    ]

    SUFFIXES = [
        # Object markers
        "ኝ", "ህ", "ሽ", "ችሁ", "ኟቸው", "ዋት", "አቸው", "አችኋል", 
        "ኧቸው", "ኣቸው", "ኣችሁ", "ኣችኋል", "ዎት", "ዎች", "ዎችን",
        
        # Tense/aspect markers
        "አለ", "አል", "ኣል", "ኣለ", "ኧል", "ኧለ", "አችሁ", "አችኋል",
        "ኧችሁ", "ኧችኋል", "አሉ", "ኧሉ", "ኣሉ", "አችሁ", "ኧሁ", "ኧህ", 
        "ኧሽ", "ኧን", "ኧ", "ኦ", "ኡ", "ኢ", "ኤ",
        
        # Verb suffixes
        "ነው", "ነበር", "ኧው", "ኧዎ", "ኧና", "ኧም", "ኧኛ", "ኧዎች", "ኧዋ", 
        "ኧኣት", "ኣት", "ኧለች", "ኣለች", "ኣለሁ", "ኧለሁ",
        
        # Nominal suffixes
        "ች", "ው", "ዎ", "ና", "ት", "ም", "ኛ", "ዎች", "ዋ", "ዋል", "ን", "ቸው"
    ]

    # Words exempt from stemming
    PROTECTED_WORDS = {
           "ውስጥ", "ላይ", "በላይ", "ከታች", "ፊት", "ኋላ", "ትርጉም", 
        "ስም", "ስራ", "ቤት", "አባት", "እናት", "ልጅ", "ወንድ", "ሴት",
        "እኔ", "አንተ", "አንቺ", "እሱ", "እሷ", "እኛ", "እናንተ", "እሳቸው",
        "ኢትዮጵያ", "አዲስ", "አበባ", "ሕግ", "ፍትህ", "ዴሞክራሲ"
    }

    def __init__(self):
        # Sort by length for longest-match-first
        self.PREFIXES.sort(key=len, reverse=True)
        self.SUFFIXES.sort(key=len, reverse=True)

    def stem_amharic(self, word):
        """Stem Amharic words using morphological analysis"""
        # Preserve protected words and short words
        if word in self.PROTECTED_WORDS or len(word) <= 3:
            return word
            
        original = word
        stem = word
        
        # Remove prefixes (max 2 iterations)
        for _ in range(2):
            removed = False
            for prefix in self.PREFIXES:
                if stem.startswith(prefix) and len(stem) > len(prefix) + 2:
                    stem = stem[len(prefix):]
                    removed = True
                    break
            if not removed:
                break
        
        # Remove suffixes (max 2 iterations)
        for _ in range(2):
            removed = False
            for suffix in self.SUFFIXES:
                if stem.endswith(suffix) and len(stem) > len(suffix) + 2:
                    stem = stem[:-len(suffix)]
                    removed = True
                    break
            if not removed:
                break
        
        # Fallback to original if stem is invalid
        return stem if len(stem) >= 3 else original

