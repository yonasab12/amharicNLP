

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

**amharicNLP** solves this challenge with a full, language-specific preprocessing pipeline.

---

# ⚙️ **What Is amharicNLP?**

`amharicNLP` is a modular Python package built for end-to-end Amharic text preprocessing.

### 🧩 It includes six core components:

1. **Cleaner** – Removes HTML, emojis, numbers & noise
2. **Normalizer** – Fixes inconsistencies in characters & punctuation
3. **Tokenizer** – Splits text into meaningful tokens
4. **Stopword Processor** – Removes common filler words
5. **Lemmatizer** – Converts words to their base dictionary form
6. **Stemmer** – Reduces words to their root for ML tasks

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

# 🧭 **Module Summary**

| Step | Module                       | Purpose                                 |
| ---- | ---------------------------- | --------------------------------------- |
| 1    | **AmharicCleaner**           | Removes noise, HTML, punctuation errors |
| 2    | **AmharicNormalizer**        | Standardizes characters & spacing       |
| 3    | **AmharicWordTokenizer**     | Splits text into tokens                 |
| 4    | **AmharicStopwordProcessor** | Removes common stopwords                |
| 5    | **AmharicLemmatizer**        | Finds base word form                    |
| 6    | **AmharicStemmer**           | Extracts root word                      |

---

# 🚀 **Final Thoughts**

`amharicNLP` bridges the gap between AI and one of Africa’s most important Semitic languages.
With only a few lines of code, you can prepare Amharic data for machine learning, deep learning, and linguistic analysis.

> **“By teaching computers to understand Amharic, we make technology speak our language.”** 🇪🇹💻

---

# ✍️ **Author**

**👤 Yonas Aebebe**

Exploring Amharic NLP, machine learning, and AI tools for Ethiopian languages.
GitHub: **yonasab12**






