

# 🇪🇹 **Amharic NLP Toolkit**

**Amharic NLP Toolkit** is a lightweight, powerful, and easy-to-use Natural Language Processing (NLP) toolkit designed specifically for the **Amharic language**.
It provides complete tools for Amharic text preprocessing, cleaning, tokenization, normalization, stopword removal, stemming, lemmatization, and sentiment analysis.

Perfect for **machine learning**, **deep learning**, **LLMs**, **AI projects**, and any Ethiopian language application.

---

# 🌍 **Why Amharic Needs Its Own NLP Toolkit**

Amharic — Ethiopia’s official language — is **morphologically rich** and **syntactically complex**.

A single Amharic word can contain:

✔️ Subject
✔️ Tense
✔️ Negation
✔️ Verb root
✔️ Suffix

Example:
**“አልሄደም”** = negation + verb root + suffix.

Tools built for English (NLTK, SpaCy) cannot correctly handle:

* Fidel script
* Complex morphology
* Combined affixes
* Amharic punctuation
* Unicode inconsistencies

**amharicNLP** solves this challenge with a full, language-specific preprocessing pipeline, plus an optional layer of pretrained transformer models for sentiment, NER, and embeddings.

---

# ⚡ **Quick Start**

```python
from amharicNLP import AmharicNLP

nlp = AmharicNLP()                       # rule-based only, zero extra deps
doc = nlp("በጣም ጥሩ ፊልም ነበር።")
doc.tokens   # ['በጣም', 'ጥሩ', 'ፊልም', 'ነበር', '።']
doc.stems    # rule-based affix-stripped stems

nlp = AmharicNLP(neural=True)            # needs: pip install amharicNLP[neural]
doc = nlp("በጣም ጥሩ ፊልም ነበር።")
doc.sentiment   # {'label': 'positive', 'score': 0.97}
doc.entities    # [{'entity_group': 'ORG', ...}, ...]
doc.embedding   # list[float], for semantic search / similarity

doc.subword_tokens   # BPE subword tokens, for feeding an LLM (see below)
doc.pos_tags         # [('እኔ','PRON'), ('ኢትዮጵያዊ','ADJ'), ('ነኝ','AUX'), ('።','PUNCT')] -- rule-based, see below
```

`AmharicNLP()` is the single entry point; the sections below cover the individual components it's built from, if you want to use them directly.

---

# ✂️ **Two Different Tokenizers, Two Different Jobs**

This package now includes **both** kinds of tokenizer, because they solve different problems:

| | `AmharicWordTokenizer` (ours) | `AmharicSubwordTokenizer` (wraps [sefineh-ai/Amharic-Tokenizer](https://github.com/sefineh-ai/Amharic-Tokenizer)) |
|---|---|---|
| Output | Whole words: `['ተማሪዎች', 'ናቸው']` | BPE fidel-decomposed pieces: `['ተአመኣረኢወኦቸእ<eow>', ...]` |
| Use it for | Stemming, stopword removal, the classic NLP pipeline | Feeding text into an LLM/embedding model, counting tokens for a context window |
| Dependencies | None (stdlib) | `amharic-tokenizer` (installed automatically, ~10MB, no torch) |

```python
from amharicNLP import AmharicSubwordTokenizer

sub = AmharicSubwordTokenizer()
sub.tokenize("የካቲት ወር ላይ ተማሪዎች ናቸው።")
sub.token_count("...")   # how many tokens an LLM would see
```

`AmharicSubwordTokenizer.tokenize()` → `.detokenize()` round-trips exactly. `.encode()` → `.decode()`
can drop trailing punctuation — a quirk in the underlying package, not this wrapper; prefer
tokenize/detokenize if you need an exact round trip.

---

# ⚙️ **What Is amharicNLP?**

`amharicNLP` is a modular Python package built for end-to-end Amharic text preprocessing.

### 🧩 It includes six core components:

1. **Cleaner** – Removes HTML, emojis, numbers & noise
2. **Normalizer** – Fixes inconsistencies in characters & punctuation
3. **Tokenizer** – Splits text into meaningful tokens
4. **Stopword Processor** – Removes common filler words
5. **Lemmatizer** – Related-term/synonym lookup (see note below — not inflection→lemma)
6. **Stemmer** – Rule-based affix stripping: strips known Amharic prefixes/suffixes and the አል...ም negation circumfix down to an approximate root

> **Note on "lemmatization":** the bundled `data_lemma` table maps words to *semantically related terms*
> (e.g. "አጭር" (short) → ["ድንክ", "ድውይ", "ከረከንዳ"]), not to inflected forms of the same word. `AmharicLemmatizer`
> is therefore a related-term lookup, not true morphological lemmatization. For actual morphological
> normalization, use `AmharicStemmer` — it does real prefix/suffix stripping (negation, tense/aspect
> prefixes, plural/possessive suffixes), not just related-word lookup.

---

# 🧠 **Optional AI Layer (sentiment, NER, embeddings)**

The rule-based tools above have zero third-party dependencies. For tasks that genuinely need a trained
model — sentiment, named entity recognition, semantic similarity — install the `neural` extra:

```bash
pip install amharicNLP[neural]
```

This wires in small, free, Amharic-specific transformer models already pretrained and published on the
Hugging Face Hub (not trained by this package):

- `rasyosef/bert-medium-amharic-finetuned-sentiment` — sentiment
- `rasyosef/bert-medium-amharic-finetuned-ner` — named entity recognition
- `rasyosef/bert-medium-amharic` — base model, used for sentence embeddings

They were pretrained from scratch on ~290M tokens of Amharic text and run fine on CPU. Full credit to
[rasyosef/bert-amharic](https://github.com/rasyosef/bert-amharic) for training and publishing them.

```python
from amharicNLP.neural import AmharicNeuralPipeline

ai = AmharicNeuralPipeline()
ai.sentiment("አሪፍ ፊልም ነው።")                          # {'label': 'positive', 'score': 0.98}
ai.entities("ጠቅላይ ሚኒስትር አብይ አህመድ ትናንት ተናገሩ።")        # [{'entity_group': 'PER', 'word': 'አብይ አህመድ', ...}]
ai.similarity("ጥሩ ፊልም ነው።", "በጣም ደስ የሚል ፊልም ነው።")     # cosine similarity, 0.0-1.0
```

If you need something the base models don't cover (POS tagging, dependency parsing, a bigger LLM),
also worth a look: [HornMorpho](https://github.com/hltdi/HornMorpho) (finite-state morphological
analyzer/generator for Amharic, Tigrinya, Oromo) for real morphological analysis beyond what the
rule-based stemmer above approximates, and
[EthioNLP/Amharic-LLAMA](https://huggingface.co/EthioNLP/Amharic-LLAMA-all-data) for generative tasks.

---

# 📦 **Installation**

### **Option 1: Install from PyPI (Recommended)**

```bash
pip install amharicNLP
```

### **Option 2: Install Latest Development Version**

```bash
git clone https://github.com/yonasab12/amharicNLP.git
cd amharicNLP
pip install .
```

---

# 🧪 **Full Demo: End-to-End Amharic Text Preprocessing**

```python
from amharicNLP.resources.cleaner import AmharicCleaner
from amharicNLP.resources.normalizer import AmharicNormalizer
from amharicNLP.resources.lemmatizer import AmharicLemmatizer
from amharicNLP.resources.stemmer import AmharicStemmer
from amharicNLP.resources.stopwrod import AmharicStopwordProcessor
from amharicNLP.resources.tokenizer import AmharicWordTokenizer

sample_text = "በአገራችን ኢትዮጵያ <h1/> ላ ያሉ ተማሪዎች በትምህርት ላይ ትኩረት ማድረግ አለባቸው። 123 ቁጥር! በላይ ዘለቀ ጀግና የኢትዮጵያ አርበኛ ነበር።"
```

---

## 🧹 **Step 1: Cleaning**

```python
cleaner = AmharicCleaner()
cleaned_html = cleaner.remove_html(sample_text)
cleaned_text = cleaner.remove_noise(cleaned_html)
print(cleaned_text)
```

**Output**

```
በአገራችን ኢትዮጵያ ላ ያሉ ተማሪዎች በትምህርት ላይ ትኩረት ማድረግ አለባቸው። ቁጥር በላይ ዘለቀ ጀግና የኢትዮጵያ አርበኛ ነበር።
```

✔️ HTML removed
✔️ Numbers & non-Amharic characters cleaned

---

## 🔤 **Step 2: Normalization**

```python
normalizer = AmharicNormalizer()
text1 = normalizer.normalize_amharic_chars(cleaned_text)
text2 = normalizer.normalize_punctuation_spacing(text1)
text3 = normalizer.expand_abbreviations(text2)
print(text3)
```

✔️ Standardized characters
✔️ Clean punctuation spacing

---

## 🪶 **Step 3: Stopword Removal**

```python
stopword_processor = AmharicStopwordProcessor()
filtered_text = stopword_processor.remove(text3)
print(filtered_text)
```

✔️ Removes high-frequency filler words

---

## 📖 **Step 4: Lemmatization**

```python
lemmatizer = AmharicLemmatizer()
lemmatized_text = lemmatizer.lemmatize(filtered_text)
print(lemmatized_text)
```

✔️ Converts words to canonical dictionary forms

---

## 🌱 **Step 5: Stemming**

```python
stemmer = AmharicStemmer()
stemmed = [stemmer.stemaize(word) for word in filtered_text]
print(stemmed)
```

✔️ Ideal for ML pipelines (text clustering, topic modeling)

---

# 🧠 **Why This Matters**

`amharicNLP` significantly improves NLP performance by:

* Cleaning & normalizing messy text
* Reducing vocabulary sparsity
* Preparing text for downstream tasks like:
  ✔ Sentiment analysis
  ✔ Text classification
  ✔ POS tagging
  ✔ Named Entity Recognition (NER)
  ✔ Language modeling

---

# 🏷️ **Rule-Based POS Tagging (no model, no dependency)**

`AmharicPOSTagger` tags each token with a reduced Universal Dependencies tag
(NOUN, VERB, AUX, ADJ, ADV, PRON, DET, ADP, CCONJ, SCONJ, NUM, PROPN, INTJ,
PUNCT), using only hand-curated lexicons (pronouns, conjunctions, auxiliaries,
numbers) and the same morphological patterns the stemmer already knows about
(verb prefixes, the አል...ም negation circumfix, adjectival suffixes like -ኛ/-ዊ).
No training data, no external model.

```python
from amharicNLP import AmharicPOSTagger
from amharicNLP.resources.tokenizer import AmharicWordTokenizer

tagger = AmharicPOSTagger()
tokens = AmharicWordTokenizer().tokenize("እኔ ኢትዮጵያዊ ነኝ።")
tagger.tag(tokens)
# [('እኔ', 'PRON'), ('ኢትዮጵያዊ', 'ADJ'), ('ነኝ', 'AUX'), ('።', 'PUNCT')]
```

or via `amharicnlp pos --text "..."` on the command line, or `doc.pos_tags` from `AmharicNLP()`.

**Be honest about what this is:** closed-class tags (pronouns, conjunctions, numbers,
punctuation) are reliable lexicon lookups but don't generalize past the words listed.
VERB/ADJ detection is a morphological *guess* — it will misfire on words that only
coincidentally match a verb-like prefix/suffix pattern, and anything unmatched
defaults to NOUN (the most common open class) rather than "unknown". Bare past-tense
verb roots (e.g. "ሄደ") carry no prefix or suffix at all, so they're only caught via
a small hand-curated list, not the general pattern — extend `BARE_VERB_ROOTS` in
`resources/pos_tagger.py` for your domain's common verbs.

Tested against 30 real, varied sentences (not just the unit-test examples) — two
concrete findings from that, both since fixed: the negation circumfix's very common
"አላ...ም" variant (e.g. "አላነበብኩም", didn't read) wasn't recognized at all, and
past-tense verbs conjugated with a suffix only, no prefix (e.g. "አበሰለች", she cooked
— the most common declarative verb form in Amharic) were structurally excluded by
a prefix-AND-suffix check. Widening the suffix match to catch those also broke
common plural nouns like "ሰራተኞች" (workers) and object-marked nouns like "ጨዋታውን"
(the game) — ች/ን are shared by both verb agreement and noun morphology — so the fix
now explicitly excludes -ኞች/-ዎች/-ኦች/-ውን/-ዋን before applying the suffix-only check.
Some verb endings (-ጉ, -ተ, -በ, as in "አደረጉ" they did, "ተጫወተ" he played, "ዘነበ" it
rained) are still deliberately NOT covered, because a first attempt at adding them
collided with ordinary nouns the same way — closing that gap needs either a labeled
corpus to check against or per-ending collision analysis this module doesn't have.
This is a rough first pass, not a validated tagger — if you need one that's actually
benchmarked against labeled data, look at fine-tuning a model on something like
MasakhaPOS, or at [HornMorpho](https://github.com/hltdi/HornMorpho)'s real
morphological analyzer.

---

# 🧭 **Module Summary**

| Step | Module                       | Purpose                                 |
| ---- | ---------------------------- | --------------------------------------- |
| 1    | **AmharicCleaner**           | Removes noise, HTML, punctuation errors |
| 2    | **AmharicNormalizer**        | Standardizes characters & spacing       |
| 3    | **AmharicWordTokenizer**     | Splits text into tokens                 |
| 4    | **AmharicStopwordProcessor** | Removes common stopwords                |
| 5    | **AmharicLemmatizer**        | Related-term/synonym lookup             |
| 6    | **AmharicStemmer**           | Rule-based affix stripping (real root extraction) |
| 7    | **AmharicNLP**                | Unified pipeline combining 1-6, plus optional `neural=True` for sentiment/NER/embeddings |
| 8    | **AmharicNeuralPipeline**     | Optional: sentiment, NER, embeddings via pretrained transformer models (`pip install amharicNLP[neural]`) |

---

# 🖥️ **Command Line**

```bash
amharicnlp tokenize --text "ሰላም ለዓለም"
amharicnlp normalize --text "ኣብዚ ሠላም"
```

---

# 🚀 **Final Thoughts**

`amharicNLP` bridges the gap between AI and one of Africa’s most important Semitic languages.
With only a few lines of code, you can prepare Amharic data for machine learning, deep learning, and linguistic analysis.

> **“By teaching computers to understand Amharic, we make technology speak our language.”** 🇪🇹💻

---

# ✍️ **Author**

**👤 Yonas Abebe**

Exploring Amharic NLP, machine learning, and AI tools for Ethiopian languages.
GitHub: **yonasab12**






