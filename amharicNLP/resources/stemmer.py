import re
import ast
from .amharic_stopword import data as stop_words

class AmharicLemmatizer:
    def __init__(self, wordnet_data=None):
        if wordnet_data is None:
            # Load default WordNet data from the package
            import amharicNLP
            import os
            
            # Get the directory of the amharicNLP package
            package_dir = os.path.dirname(amharicNLP.__file__)
            wordnet_path = os.path.join(package_dir, 'data_lemma.py')
            
            # Read and parse the Python file
            with open(wordnet_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
            # Parse the Python file to extract the wordnet_data variable
            parsed = ast.parse(file_content)
            for node in parsed.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'wordnet_data':
                            # Use ast.literal_eval to safely evaluate the expression
                            self.wordnet_data = ast.literal_eval(node.value)
                            break
                    else:
                        continue
                    break
            else:
                self.wordnet_data = []
        else:
            self.wordnet_data = wordnet_data

    def lemmatize(self, word):
        for entry in self.wordnet_data:
            if word in entry["example"]:
                return entry["lemma"]
        return word

class AmharicStemmerprocess:
    def __init__(self):
        self.PROTECTED_WORDS = {
            "ውስጥ", "ላይ", "በላይ", "ከታች", "ፊት", "ኋላ", "ትርጉም", 
            "ስም", "ስራ", "ቤት", "አባት", "እናት", "ልጅ", "ወንድ", "ሴት",
            "እኔ", "አንተ", "አንቺ", "እሱ", "እሷ", "እኛ", "እናንተ", "እሳቸው",
            "ኢትዮጵያ", "አዲስ አበባ", "ሕግ", "ፍትህ", "ዴሞክራሲ"
        }

        self.phonological_rules = [
            (r'ኧ', ''),
            (r'[ዋዉዊዌ]', 'ም'),
            (r'ሽ', 'ስ'),
            (r'ች', 'ት'),
            (r'ኝ', 'ን'),
            (r'ጅ', 'ግ'),
            (r'ዥ', 'ድ'),
            (r'ጭ', 'ጥ'),
            (r'([ሃሐሓልምን])$', r'\1ም'),
            (r'(\w)\1', r'\1')
        ]

        self.prefix_rules = [
            (r'^(እንደ|ያለ|እስከ|በስተ|በኩል|በውስጥ|ከውስጥ|ለምሳሌ|ከላይ|በላይ)', ''),
            (r'^(እየ|ይየ|አስተ|አስ|ተስ|አል|አት|አይ|ይተ)', ''),
            (r'^(ይ|ት|እ|ኢ|ል|ተ|አ|ስ|የ|ን|ታ|አም|ተም)', '')
        ]

        self.suffix_rules = [
            (r'(ኟቸው|ዋት|አቸው|አችኋል|ኧቸው|ኣቸው|ኣችሁ|ኣችኋል|ዎት|ዎች|ዎችን)$', ''),
            (r'(አለ|ኣል|ኣለ|ኧል|ኧለ|አችሁ|አችኋል|ኧችሁ|ኧችኋል|አሉ|ኧሉ|ኣሉ)$', ''),
            (r'(ነው|ነበር|ኧው|ኧዎ|ኧና|ኧም|ኧኛ|ኧዎች|ኧዋ|ኧኣት|ኣት|ኧለች|ኣለች|ኣለሁ|ኧለሁ)$', ''),
            (r'(ች|ው|ዎ|ና|ት|ም|ኛ|ዎች|ዋ|ዋል|ን|ቸው)$', '')
        ]

        self.root_patterns = [
            (r'(ላ|ማ|ና|ቻ|ያ|ቃ|ሳ|ጋ|ዳ|ጣ|ፋ)$', lambda m: m.group(1)[0]),
            (r'[ላማናቻያቃሳጋዳጣፋ]$', 'ም')
        ]

    def stem(self, word):
        if word in self.PROTECTED_WORDS or len(word) <= 3:
            return word
        stem = self._apply_rules(word, self.phonological_rules)
        stem = self._apply_rules(stem, self.prefix_rules)
        stem = self._apply_rules(stem, self.suffix_rules)
        stem = self._apply_rules(stem, self.root_patterns)
        stem = self._normalize(stem)
        return stem if len(stem) >= 2 else word

    def _apply_rules(self, word, rules):
        for pattern, replacement in rules:
            if callable(replacement):
                word = re.sub(pattern, replacement, word)
            else:
                word = re.sub(pattern, replacement, word)
        return word

    def _normalize(self, stem):
        stem = re.sub(r'ኧ', '', stem)
        if len(stem) > 1 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem

class AmharicStemmer:
    def __init__(self, wordnet_data=None):
        self.lemmatizer = AmharicLemmatizer(wordnet_data)
        self.stemmer = AmharicStemmerprocess()

    def stemaize(self, word):
        # First try lemmatization
        lemma = self.lemmatizer.lemmatize(word)
        
        # If lemmatization didn't change the word, apply stemming
        if lemma == word:
            return self.stemmer.stem(word)
        return lemma

    def process_text(self, text):
        tokens = text.split()
        cleaned = [word for word in tokens if word not in stop_words]
        return [self.process_word(word) for word in cleaned]
