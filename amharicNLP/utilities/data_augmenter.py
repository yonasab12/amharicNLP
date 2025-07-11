import random


from amharicNLP.resources.tokenizer import AmharicWordTokenizer
from amharicNLP.resources.cleaner import AmharicCleaner
from amharicNLP.resources.normalizer import AmharicNormalizer
from amharicNLP.resources.lemmatizer import AmharicLemmatizer
from amharicNLP.resources.stemmer import AmharicStemmer        
# Initialize class-based components
tokenizer = AmharicWordTokenizer()
cleaner = AmharicCleaner()

SYNONYMS = {
    "እንዴት": ["እንደምን", "በምን አይነት"],
    "ነህ": ["ነሽ", "ናችሁ"],
    "ደስ": ["ሐሰት", "ደስታ"],
    "ብሎኛል": ["አደሰኝ", "ደስ አለኝ"]
}

def random_deletion(words, p=0.2):
    if len(words) < 2:
        return words
    return [w for w in words if random.random() > p] or [random.choice(words)]

def synonym_replacement(words, n=1):
    new_words = words.copy()
    indices = [i for i, w in enumerate(new_words) if w in SYNONYMS]
    random.shuffle(indices)
    
    for i in indices[:min(n, len(indices))]:
        new_words[i] = random.choice(SYNONYMS[new_words[i]])
    return new_words

def random_swap(words, n=1):
    new_words = words.copy()
    for _ in range(min(n, len(words) // 2)):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def augment_sentence(sentence, aug_prob=0.3):
    words = tokenizer.tokenize(sentence)
    if random.random() < aug_prob:
        words = random_deletion(words)
    if random.random() < aug_prob:
        words = synonym_replacement(words, n=1)
    if random.random() < aug_prob:
        words = random_swap(words)
    return tokenizer.detokenize(words)

# Preserving your original example usage
if __name__ == "__main__":
    sentence = "እንዴት ነህ ደስ ብሎኛል"
    print("Original:", sentence)
    
    words = tokenizer.tokenize(sentence)
    
    print("Random Deletion:", tokenizer.detokenize(random_deletion(words, p=0.3)))
    print("Synonym Replacement:", tokenizer.detokenize(synonym_replacement(words, n=2)))
    print("Random Swap:", tokenizer.detokenize(random_swap(words, n=2)))
    print("Remove Diacritics:", cleaner.remove_diacritics(sentence))