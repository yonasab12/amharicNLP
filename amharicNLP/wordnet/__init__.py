import json
from pathlib import Path
from ..resources.lemmatizer import AmharicLemmatizer  

# Load WordNet data
_wordnet_path = Path(__file__).parent / "amh_wordnet.json"
with open(_wordnet_path, encoding='utf-8') as f:
    wordnet_data = json.load(f)

# Initialize lemmatizer (without stemmer)
lemmatizer = AmharicLemmatizer(wordnet_data)

# Make available at package level
__all__ = ['wordnet_data', 'lemmatizer']