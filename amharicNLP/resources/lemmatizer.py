import os
import json
import amharicNLP
print(amharicNLP.__file__)
class AmharicLemmatizer:
    def __init__(self, wordnet_path=None):
        # Handle direct dictionary input
        if isinstance(wordnet_path, dict):
            self.wordnet = wordnet_path
        else:
            # Handle case where wordnet_path might be a list
            if isinstance(wordnet_path, list) and wordnet_path:
                wordnet_path = wordnet_path[0]  # Take first element if it's a non-empty list
            
            if wordnet_path is None:
                current_dir = os.path.dirname(__file__)
                wordnet_path = os.path.join(current_dir, "..", "wordnet", "amh_wordnet.json")
            
            # Validate path type
            if not isinstance(wordnet_path, (str, bytes, os.PathLike)):
                # Get the actual type name for better error message
                actual_type = type(wordnet_path).__name__
                raise TypeError(f"Invalid path type: {actual_type}. Expected string, path-like object, or dictionary.")
            
            if not os.path.exists(wordnet_path):
                raise FileNotFoundError(f"WordNet file not found at {wordnet_path}")
            
            self.wordnet = self.load_wordnet(wordnet_path)
        
        self.lemma_map = {}
        self.fallback_map = {}
        self.build_lemma_maps()
    
    def load_wordnet(self, path):
        # If we already have dictionary data, return it directly
        if isinstance(path, dict):
            return path
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def build_lemma_maps(self):
        # Ensure we're working with a list of synsets
        if isinstance(self.wordnet, dict) and "synsets" in self.wordnet:
            wordnet_data = self.wordnet["synsets"]
        elif isinstance(self.wordnet, list):
            wordnet_data = self.wordnet
        else:
            # Handle unexpected structure
            wordnet_data = []
        
        for synset in wordnet_data:
            # Skip synsets without 'lemmas' key
            if "lemmas" not in synset or not synset["lemmas"]:
                continue
                
            base_lemma = synset["lemmas"][0]
            pos_tag = synset.get("pos", None)
            
            for lemma in synset["lemmas"]:
                normalized = lemma.lower()
                
                # Add to POS-specific map if POS is available
                if pos_tag:
                    pos_key = (normalized, pos_tag)
                    if pos_key not in self.lemma_map:
                        self.lemma_map[pos_key] = base_lemma
                
                # Add to fallback map
                if normalized not in self.fallback_map:
                    self.fallback_map[normalized] = base_lemma
    
    def lemmatize(self, word, pos=None):
        normalized = word.lower()
        
        if pos is not None:
            # Try POS-specific lookup
            pos_key = (normalized, pos)
            if pos_key in self.lemma_map:
                return self.lemma_map[pos_key]
        
        # Try fallback (POS=None)
        if normalized in self.fallback_map:
            return self.fallback_map[normalized]
        
        return word