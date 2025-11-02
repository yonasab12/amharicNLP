import re
from . import data_lemma
from .amharic_stopword import data as stop_words

# ----------------- Lemmatizer -----------------
class AmharicLemmatizer:
    def __init__(self, wordnet_data=None):
        self.wordnet_data = wordnet_data or data_lemma.data

    def lemmatize(self, word):
        for entry in self.wordnet_data:
            if word in entry["example"]:
                return entry["lemma"]
        return word


# ----------------- Stemmer Process -----------------
class AmharicStemmerprocess:
    def __init__(self):
        self.PROTECTED_WORDS = {
            "ኢትዮጵያ", "አዲስ", "አበባ", "ሰላም", "ፍቅር", "ቤት", "ስራ", "ልጅ",
            "እኔ", "እኛ", "አንተ", "አንቺ", "እሱ", "እሷ", "እናንተ", "እነሱ"
        }

       # Amharic alphabet variations for normalization
        self.amharic_alphabet = [
            {"base": "ሀ", "variations": ["ሀ", "ሁ", "ሂ", "ሃ", "ሄ", "ህ", "ሆ"]},
            {"base": "ለ", "variations": ["ለ", "ሉ", "ሊ", "ላ", "ሌ", "ል", "ሎ"]},
            {"base": "ሐ", "variations": ["ሐ", "ሑ", "ሒ", "ሓ", "ሔ", "ሕ", "ሖ"]},
            {"base": "መ", "variations": ["መ", "ሙ", "ሚ", "ማ", "ሜ", "ም", "ሞ"]},
            {"base": "ሠ", "variations": ["ሠ", "ሡ", "ሢ", "ሣ", "ሤ", "ሥ", "ሦ"]},
            {"base": "ረ", "variations": ["ረ", "ሩ", "ሪ", "ራ", "ሬ", "ር", "ሮ"]},
            {"base": "ሰ", "variations": ["ሰ", "ሱ", "ሲ", "ሳ", "ሴ", "ስ", "ሶ"]},
            {"base": "ሸ", "variations": ["ሸ", "ሹ", "ሺ", "ሻ", "ሼ", "ሽ", "ሾ"]},
            {"base": "ቀ", "variations": ["ቀ", "ቁ", "ቂ", "ቃ", "ቄ", "ቅ", "ቆ"]},
            {"base": "በ", "variations": ["በ", "ቡ", "ቢ", "ባ", "ቤ", "ብ", "ቦ"]},
            {"base": "ተ", "variations": ["ተ", "ቱ", "ቲ", "ታ", "ቴ", "ት", "ቶ"]},
            {"base": "ቸ", "variations": ["ቸ", "ቹ", "ቺ", "ቻ", "ቼ", "ች", "ቾ"]},
            {"base": "ኀ", "variations": ["ኀ", "ኁ", "ኂ", "ኃ", "ኄ", "ኅ", "ኆ"]},
            {"base": "ነ", "variations": ["ነ", "ኑ", "ኒ", "ና", "ኔ", "ን", "ኖ"]},
            {"base": "ኘ", "variations": ["ኘ", "ኙ", "ኚ", "ኛ", "ኜ", "ኝ", "ኞ"]},
            {"base": "አ", "variations": ["አ", "ኡ", "ኢ", "ኣ", "ኤ", "እ", "ኦ"]},
            {"base": "ከ", "variations": ["ከ", "ኩ", "ኪ", "ካ", "ኬ", "ክ", "ኮ"]},
            {"base": "ኸ", "variations": ["ኸ", "ኹ", "ኺ", "ኻ", "ኼ", "ኽ", "ኾ"]},
            {"base": "ወ", "variations": ["ወ", "ዉ", "ዊ", "ዋ", "ዌ", "ው", "ዎ"]},
            {"base": "ዐ", "variations": ["ዐ", "ዑ", "ዒ", "ዓ", "ዔ", "ዕ", "ዖ"]},
            {"base": "ዘ", "variations": ["ዘ", "ዙ", "ዚ", "ዛ", "ዜ", "ዝ", "ዞ"]},
            {"base": "ዠ", "variations": ["ዠ", "ዡ", "ዢ", "ዣ", "ዤ", "ዥ", "ዦ"]},
            {"base": "የ", "variations": ["የ", "ዩ", "ዪ", "ያ", "ዬ", "ይ", "ዮ"]},
            {"base": "ደ", "variations": ["ደ", "ዱ", "ዲ", "ዳ", "ዴ", "ድ", "ዶ"]},
            {"base": "ጀ", "variations": ["ጀ", "ጁ", "ጂ", "ጃ", "ጄ", "ጅ", "ጆ"]},
            {"base": "ገ", "variations": ["ገ", "ጉ", "ጊ", "ጋ", "ጌ", "ግ", "ጎ"]},
            {"base": "ጠ", "variations": ["ጠ", "ጡ", "ጢ", "ጣ", "ጤ", "ጥ", "ጦ"]},
            {"base": "ጨ", "variations": ["ጨ", "ጩ", "ጪ", "ጫ", "ጬ", "ጭ", "ጮ"]},
            {"base": "ጰ", "variations": ["ጰ", "ጱ", "ጲ", "ጳ", "ጴ", "ጵ", "ጶ"]},
            {"base": "ጸ", "variations": ["ጸ", "ጹ", "ጺ", "ጻ", "ጼ", "ጽ", "ጾ"]},
            {"base": "ፀ", "variations": ["ፀ", "ፁ", "ፂ", "ፃ", "ፄ", "ፅ", "ፆ"]},
            {"base": "ፈ", "variations": ["ፈ", "ፉ", "ፊ", "ፋ", "ፌ", "ፍ", "ፎ"]},
            {"base": "ፐ", "variations": ["ፐ", "ፑ", "ፒ", "ፓ", "ፔ", "ፕ", "ፖ"]},
            {"base": "ቨ", "variations": ["ቨ", "ቩ", "ቪ", "ቫ", "ቬ", "ቭ", "ቮ"]}
        ]

        self.normalize_map = {}
        for group in self.amharic_alphabet:
            for var in group["variations"]:
                self.normalize_map[var] = group["base"]

        # Allowed index rule (1,2,4)
        self.allowed_indexes = {1, 2, 4}

    def _clean_woche_suffix(self, word):
        """Remove ዎች only if allowed pattern"""
        if not word.endswith("ዎች"):
            return word
        base = word[:-2]
        if not base:
            return word
        last_char = base[-1]
        for group in self.amharic_alphabet:
            if last_char in group["variations"]:
                idx = group["variations"].index(last_char)
                if idx in self.allowed_indexes:
                    return base
        return word

    def normalize(self, word):
        return ''.join(self.normalize_map.get(ch, ch) for ch in word)

    def stem(self, word):
        """Normalization + ዎች rule"""
        if isinstance(word, list):
            return [self.stem(w) for w in word]
        if word in self.PROTECTED_WORDS or len(word) <= 2:
            return word

        stem = self.normalize(word)
        stem = self._clean_woche_suffix(stem)
        stem = self.normalize(stem)
        return stem if len(stem) >= 2 else word


# ----------------- Combined Rule: Lemmatize → Stem -----------------
class AmharicStemmer:
    def __init__(self, wordnet_data=None):
        self.lemmatizer = AmharicLemmatizer(wordnet_data)
        self.stemmer = AmharicStemmerprocess()

    def stemaize(self, word):
        """
        ✅ Rule:
        1️⃣ Try lemmatization first
        2️⃣ If lemmatization fails (lemma == word), then apply stemming
        """
        lemma = self.lemmatizer.lemmatize(word)
        if lemma == word:  # Lemmatization did not change
            return self.stemmer.stem(word)
        return lemma  # Lemmatization worked → keep it

    def process_text(self, text):
        """Remove stopwords, then apply Lemmatization→Stemming rule."""
        tokens = text.split()
        cleaned = [w for w in tokens if w not in stop_words]
        return [self.stemaize(w) for w in cleaned]


# ----------------- Example Usage -----------------
if __name__ == "__main__":
    stemmer = AmharicStemmer()

    test_words = [
        "ኩዎች", "ሊዎች", "ማዎች", "ቤዎች",
        "ልዎች", "ኮዎች", "ኢትዮጵያዎች"
    ]

    print("✅ Lemmatization → Stemming Rule Applied:\n")
    for word in test_words:
        result = stemmer.stemaize(word)
        print(f"{word}  →  {result}")






