import re
import os
from pathlib import Path
from functools import lru_cache

class AmharicNormalizer:
    """Advanced Amharic text normalizer with resource-based preprocessing"""
    
    def __init__(self):
        self.resource_dir = Path(__file__).parent / "resources"
        self._validate_resources()
        
        # Load resources once during initialization
        self.short_forms = self._load_resource("short_forms.txt", delimiter="-")
        self.stopwords = set(self._load_resource("stopwords.txt"))
        self.normalization_rules = self._load_normalization_rules()

    def _validate_resources(self):
        """Ensure required resource files exist"""
        required_files = [
            "short_forms.txt",
            "stopwords.txt", 
            "chars_to_normalize.txt"
        ]
        for file in required_files:
            if not (self.resource_dir / file).exists():
                raise FileNotFoundError(f"Missing resource file: {file}")

    def _load_resource(self, filename, delimiter=None):
        """Generic resource loader with caching"""
        filepath = self.resource_dir / filename
        with open(filepath, "r", encoding="utf-8") as f:
            if delimiter:
                return dict(line.strip().split(delimiter) for line in f if line.strip())
            return [line.strip() for line in f if line.strip()]

    def _load_normalization_rules(self):
        """Compile regex patterns for character normalization"""
        rules = []
        with open(self.resource_dir / "chars_to_normalize.txt", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    pattern, replacement = line.strip().split("→")
                    rules.append(
                        (re.compile(pattern.strip()), replacement.strip())
                    )
        return rules

    @lru_cache(maxsize=1000)
    def normalize(self, text):
        """Full normalization pipeline"""
        text = self._normalize_char_level(text)
        text = self._expand_short_forms(text)
        text = self._remove_stopwords(text)
        return text

    def _normalize_char_level(self, text):
        """Apply character normalization rules"""
        for pattern, replacement in self.normalization_rules:
            text = pattern.sub(replacement, text)
        return text

    def _expand_short_forms(self, text):
        """Replace abbreviations with their full forms"""
        return " ".join(
            self.short_forms.get(word, word)
            for word in text.split()
        )

    def _remove_stopwords(self, text):
        """Filter out stopwords using set operations"""
        return " ".join(
            word for word in text.split()
            if word not in self.stopwords
        )

    def batch_normalize(self, texts):
        """Process multiple texts efficiently"""
        return [self.normalize(text) for text in texts]