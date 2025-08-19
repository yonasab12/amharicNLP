from data_lemma import data as wordnet_data
def lemma(word):
    for entry in wordnet_data:
        if word in entry["example"]:
            return entry["lemma"]
    return word