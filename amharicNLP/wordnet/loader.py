import json
import os

# Automatically find the file relative to this script
this_dir = os.path.dirname(__file__)
wordnet_path = os.path.join(this_dir, "am_wordnet.json")

def load_amharic_wordnet():
    with open(wordnet_path, encoding="utf-8") as f:
        return json.load(f)
