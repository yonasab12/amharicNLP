

# 🇪🇹 **amharicNLP**

### A real, tested NLP toolkit for Amharic — not a wrapper around English tools.

Amharic is morphologically rich in ways that break most off-the-shelf NLP tooling: a single word
can encode subject, tense, and negation at once (**አልሄደም** = negation + verb root + subject
suffix, "he did not go"), the script has no letter case, and common orthographic variants
(ሀ/ሐ/ኀ, ሰ/ሠ, አ/ዐ) need to be reconciled before anything downstream works. Tools built for
English — NLTK, spaCy — don't handle Fidel script, this morphology, or these variants correctly,
because they were never designed to.

**amharicNLP** is a dependency-light, rule-based preprocessing pipeline built specifically for
this — cleaning, normalization, tokenization, stopword removal, real morphological stemming
(prefix/suffix stripping, not lookup tables), and a rule-based POS tagger — with an optional
layer of pretrained transformer models bolted on for sentiment, NER, and embeddings when you
need them. Useful if you're training or fine-tuning an Amharic language model, building a
sentiment/NER pipeline, or doing general Amharic NLP work and don't want to hand-roll
tokenization and stemming from scratch.

**What makes this different from "just another wrapper":** every rule-based component here
was stress-tested against real, varied Amharic sentences during development — not just the
handful of examples in the unit tests — and every fix, trade-off, and remaining known gap is
documented in the code and in this README rather than papered over. If you find a case it gets
wrong, that's expected for a from-scratch rule-based system with no training corpus behind it —
open an issue with the sentence and it'll get fixed the same way the existing fixes were: found,
verified, patched, regression-tested.

---

# 🎯 **Bring Your Own Data. Bring Your Own Model.**

amharicNLP is the **Amharic data and preprocessing layer** — not a
collection of models. You bring your data, your labels, your task, your
tokenizer, your model, your framework. amharicNLP handles the Amharic
linguistic engineering in between.

```
     YOUR DATA  →  amharicNLP  →  YOUR MODEL
   (any shape)     (Amharic         (any framework)
                 preprocessing,
                  validation,
                   splitting,
                    export)
```

```python
from amharicNLP import AmharicDatasetProcessor

processor = AmharicDatasetProcessor(task="classification")

result = processor.process(
    data=my_dataframe,            # or list, dict, JSON/CSV/JSONL path, HF Dataset
    text_column="amharic_text",   # your column names
    label_column="category",      # your labels
)

result.to_dataframe()       # → pandas
result.to_hf_dataset()      # → Hugging Face
result.export("train.jsonl")  # → disk
result.split(stratify=True)   # → train/validation/test
```

### Three guarantees

**1. Nothing is assumed.** Your column names, your label vocabulary.
There is no built-in sentiment vocabulary, no required schema. Labels
are *discovered* from your data:

```python
result.labels      # ['ጥያቄ', 'ቅሬታ', 'ጥቆማ']  — whatever you actually used
result.label2id    # {'ጥያቄ': 0, 'ቅሬታ': 1, 'ጥቆማ': 2}
```

**2. Nothing destructive happens by default.** Stemming, stopword
removal, and tokenizing are **off** unless you switch them on — they
help a bag-of-words classifier and actively damage an LLM corpus or an
ASR transcript. Results go to *new* columns, so your original text is
always recoverable:

```python
processor = AmharicDatasetProcessor(
    normalize=True, clean=True,      # on by default — safe, orthographic
    tokenize=False, stem=False,      # off by default — destructive
    remove_stopwords=False,
)
```

Enable a destructive step on a text-fidelity task and the package says so:

> `task='asr'` with stem enabled: these alter the text itself and
> normally degrade ASR training data…

**3. Nothing you didn't ask about is touched.** `speaker_id`, `domain`,
`audio`, annotation spans — every column you didn't name passes through
untouched.

### Supported tasks

| Task | What it prepares |
|---|---|
| `"classification"` | sentiment, topic, intent, spam, toxicity — **any** labels |
| `"ner"` | token classification, BIO/BIOES, **any** entity types |
| `"llm"` | pretraining, instruction tuning, chat, QA, generation |
| `"asr"` | speech-to-text transcript side |
| `"tts"` | text-to-speech text side |
| `"generic"` | text processing with no structural assumptions |

### Inspect before you train

```python
print(processor.inspect(my_data, text_column="text", label_column="label"))
```

```
  total samples      : 12000
  valid samples      : 11847
  duplicates         : 153
  label column       : label
      'positive': 9200 (76.7%)
      'negative': 2800 (23.3%)
  imbalance ratio    : 3.3x
  chars  avg/med/min/max : 84.2 / 71.0 / 3 / 1204
  avg Amharic ratio      : 94.1%
  warnings:
      ! 153 duplicate text(s) (1.3% of dataset).
      ! 412 sample(s) (3%) are less than 50% Amharic script
```

Problems are **reported, not silently deleted** — you decide. Pass
`drop_invalid=True` when you want filtering.

### Your framework, your choice

No heavy dependency is mandatory. The core runs on the standard library:

```bash
pip install amharicNLP          # core — works alone
pip install pandas              # only if you want DataFrames
pip install torch               # only if you train in PyTorch
pip install transformers datasets   # only if you use Hugging Face
```

Ask for something you haven't installed and you get a real answer, not a
`ModuleNotFoundError`:

> Exporting to a DataFrame needs the optional package 'pandas', which
> isn't installed.
>
>     pip install pandas
>
> amharicNLP's core preprocessing works without it — this is an optional
> integration, not a required dependency.

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

# 🔊 **Optional Speech Layer (text-to-speech, speech-to-text)**

For voice, not just text — a separate extra, since audio I/O deps (`scipy`, `librosa`) are a
different weight class from just running a text model:

```bash
pip install amharicNLP[speech]
```

```python
from amharicNLP import AmharicTTS, AmharicASR

tts = AmharicTTS()
tts.synthesize_to_file("ሰላም ለዓለም", "out.wav")     # facebook/mms-tts-amh (Meta MMS, VITS)

asr = AmharicASR()
asr.transcribe("recording.wav")                    # agkphysics/wav2vec2-large-xlsr-53-amharic
```

- **TTS**: `facebook/mms-tts-amh` — part of Meta's Massively Multilingual Speech project, actively
  maintained, in `transformers` since v4.33.
- **ASR**: `agkphysics/wav2vec2-large-xlsr-53-amharic` — wav2vec2-large-xlsr-53 fine-tuned on the
  Amharic Speech Corpus (Abate et al. 2005); reported ~26% WER / ~7% CER on the ALFFA/Amharic
  Readspeech validation set. Usable, not state of the art.
  [speechbrain/asr-wav2vec2-dvoice-amharic](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-amharic)
  (DVoice/AIOX Labs, ~25% WER / ~6.7% CER) is a close alternative if you'd rather use SpeechBrain.

**A note on what got left out, on purpose:** while researching this, a model called
`omnivoice-amharic` turned up claiming to be "the highest-quality open Amharic TTS model available
today," with an oddly polished multi-year language-rollout roadmap attached. That's marketing copy
in a model card, not a verified result — it wasn't benchmarked against anything checkable, so it
isn't wired in here. If you evaluate it yourself and it holds up, swapping the model ID in
`amharicNLP/speech.py` is a one-line change.

**Honesty check on this module specifically:** unlike the stemmer and POS tagger, this code has
*not* been run against real audio or real model weights — the environment that built it had no
network access to huggingface.co to test with. The API calls match each model's documented usage,
and constructing `AmharicTTS()`/`AmharicASR()` without the extra installed does raise a clear error
rather than fail silently (that part's tested) — but run one real synthesis and one real
transcription yourself before depending on this in anything real.

---

# 🗂️ **Preprocessing a Dataset for Voice/LLM/Sentiment Training**

If you're prepping a real dataset — crowdsourced ASR/TTS data, an Amharic LLM corpus, a
sentiment-labeled set — `AmharicDatasetPreprocessor` runs the same cleaning/normalization used
everywhere else in this package over a text column of a Hugging Face `datasets.Dataset`, and
leaves every other column (audio, speaker metadata, a label) untouched:

```bash
pip install amharicNLP[datasets]
```

```python
from amharicNLP import AmharicDatasetPreprocessor

prep = AmharicDatasetPreprocessor()
ds = prep.load("snapwre/amharic-speech")           # any HF dataset name
ds = prep.process(ds, text_column="sentence")
ds["train"][0]["sentence_clean"]                    # normalized transcript
ds["train"][0]["audio"]                             # untouched

# Same call for a different dataset's text column -- your LLM corpus or
# sentiment set, just a different column name:
ds = prep.process(prep.load("your-org/amharic-sentiment"), text_column="text")
```

Cleaning and normalization always run. Tokenizing, stopword removal, and stemming are opt-in
(`AmharicDatasetPreprocessor(tokenize=True, remove_stopwords=True, stem=True)`), since they're
right for some downstream tasks (a bag-of-words sentiment baseline) and wrong for others — ASR/TTS
training wants normalized *text* as the target, not a token list, and an LLM corpus usually wants
raw normalized text too, not stemmed or stopword-stripped.

**Verified against real data, not invented examples:** the test suite's sentences were copied
verbatim from the [snapwre/amharic-speech](https://huggingface.co/datasets/snapwre/amharic-speech)
dataset viewer (CC-BY-4.0, 16.9k Amharic speech clips with transcripts, crowdsourced) — fetched and
read directly, not assumed from a summary. That process found and fixed one real bug this way: "፡፡"
(a traditional doubled Ethiopic wordspace still common in news writing, equivalent to "።") was
splitting into two separate tokens instead of being read as one full stop. It showed up in roughly
a quarter of the sampled sentences. Also checked before recommending it: **a different-sounding
model that turned up during the same search, `omnivoice-amharic`, claiming to be "the highest-
quality open Amharic TTS available today" with a suspiciously polished multi-year roadmap, was
NOT used anywhere in this package** — that's unverified marketing copy in a model card, not a
checked result (see the Speech Layer section above).

**What hasn't been verified:** the end-to-end `.load()` → `datasets.load_dataset()` network call
against the real Hub dataset — this environment has no network access to huggingface.co. The
`.process()` mapping logic *has* been run end-to-end, locally, against a `Dataset` built from real
rows copied from the dataset viewer (transcripts + metadata columns, matching the schema exactly)
— confirmed the doubled-wordspace fix applies correctly and every non-text column passes through
untouched. Run one real `.load()` call yourself before trusting the network path.

---

# ✂️ **Sentence Segmentation**

Splitting Amharic text into sentences isn't "split on `.`" — the primary sentence-final marker is
`።` (plus the doubled-wordspace `፡፡` variant still common in news writing), and a naive splitter
breaks on abbreviations (`ዶ.ር`), decimal numbers (`3.5`), and URLs:

```python
from amharicNLP import split_sentences
# or: from amharicNLP.resources import AmharicSentenceSegmenter

split_sentences("ዶ/ር አበበ መጣ። እሱ ጥሩ ነው!")
# ['ዶ/ር አበበ መጣ።', 'እሱ ጥሩ ነው!']
```

It's also wired into the main pipeline: `AmharicNLP()("...").sentences`.

---

# 🗃️ **Corpus Processing (loading, validation, dedup)**

`AmharicCorpusProcessor` gets arbitrary raw data — a `.txt` file, a CSV export, a JSONL dump, a
pandas DataFrame — into a uniform `list[dict]` shape, then validates it: empty rows, exact and
near-duplicate records, corrupted/broken-Unicode text, and missing required fields. No extra
dependencies for txt/csv/json/jsonl; pandas/pyarrow are only imported if you call
`load_dataframe`/`load_parquet`.

```python
from amharicNLP import AmharicCorpusProcessor

proc = AmharicCorpusProcessor()
records = proc.load_jsonl("raw.jsonl")   # or load_txt / load_csv / load_json / load_dataframe
clean, report = proc.process(records, text_field="text", near_duplicate_threshold=0.9)
print(report.summary())
# total records: 1000
# empty: 12
# exact duplicates: 40
# near-duplicates: 8
# corrupt text: 3
# kept: 937/1000
```

---

# 🤖 **LLM Dataset Validation**

`AmharicLLMDatasetValidator` understands instruction (`{"instruction", "input", "output"}`),
prompt/completion, and multi-turn chat (`{"messages": [...]}` / ShareGPT-style
`{"conversations": [...]}`) example shapes, and checks the things that only make sense at that
level: broken conversation structure, empty responses, examples that are too long, duplicate
examples, broken Unicode, a response containing almost no Amharic at all, cheap
gibberish/echo-the-prompt heuristics, and train/eval leakage.

```python
from amharicNLP import AmharicLLMDatasetValidator

validator = AmharicLLMDatasetValidator(max_chars=4000)
clean, report = validator.process(examples, check_language_contamination=True)
print(report.summary())

# Separately, check a held-out eval set isn't leaking into train:
leaked_idx = validator.check_leakage(train_examples, eval_examples)
```

---

# 🎙️ **Audio Dataset Preparation (STT/TTS)**

`AmharicAudioDatasetProcessor` is the data layer in front of the ASR/TTS *models* in `speech.py`:
duration filtering, transcript normalization (reusing `AmharicNormalizer`/
`AmharicSentenceSegmenter`), and speaker-aware train/val/test splitting work with plain metadata,
no extra dependencies. Silence trimming, resampling, and audio-quality checks touch real audio
files and need `pip install amharicNLP[speech]`.

```python
from amharicNLP.audio_dataset import AmharicAudioDatasetProcessor

proc = AmharicAudioDatasetProcessor(target_sample_rate=16000)

# Metadata-only -- no extra deps:
records = proc.normalize_transcripts(records, text_field="text")
kept, dropped = proc.filter_by_duration(records, min_duration=1.0, max_duration=15.0)
split = proc.train_val_test_split(kept, group_field="speaker_id")  # keeps each speaker in one split

# Needs pip install amharicNLP[speech]:
proc.trim_silence("clip.wav", "clip_trimmed.wav")
proc.resample("clip.wav", "clip_16k.wav", target_sr=16000)
```

---

# 🗣️ **Voice AI Pipeline**

`AmharicVoicePipeline` wires together `Audio → STT → Amharic NLP → LLM → Amharic Text → TTS →
Audio`. It does **not** bundle an LLM — per this package's core philosophy, you inject one:

```python
from amharicNLP.voice_pipeline import AmharicVoicePipeline

def my_llm(transcript: str, doc) -> str:
    ...  # call whatever LLM/API you want; return Amharic response text

pipeline = AmharicVoicePipeline(llm_fn=my_llm)
result = pipeline.process_audio_turn("question.wav", output_audio_path="answer.wav")
result.transcript              # what the user said (ASR output)
result.doc                     # tokens / pos_tags / sentiment_lexicon / sentences for the transcript
result.response_text           # the LLM's (normalized) reply
result.response_audio_path     # where the synthesized reply was written

# Or use just the NLP-understanding stage, no LLM/TTS required:
doc = pipeline.understand("ይህ ጥሩ ፊልም ነበር።")
```

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
from amharicNLP.resources.stopword import AmharicStopwordProcessor
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

`amharicNLP` prepares raw Amharic text for downstream tasks by:

* Cleaning & normalizing messy text
* Reducing vocabulary sparsity (real morphological stemming, not just lowercasing)
* Producing tokens ready for:
  ✔ Sentiment analysis (via the optional neural layer)
  ✔ Text classification
  ✔ POS tagging (built in, rule-based)
  ✔ Named Entity Recognition (via the optional neural layer)
  ✔ Language model training/fine-tuning (via the subword tokenizer)

No benchmark numbers here — this is preprocessing infrastructure, not a scored model. If you run
it against your own downstream task and it measurably helps (or doesn't), that's worth an issue.

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
| 2    | **AmharicNormalizer**        | Standardizes characters & spacing; URL/email/hashtag/mention & repeated-char/punct handling |
| 3    | **AmharicWordTokenizer**     | Splits text into tokens                 |
| 4    | **AmharicSentenceSegmenter** | Splits text into sentences (abbreviation/URL/decimal-aware) |
| 5    | **AmharicStopwordProcessor** | Removes common stopwords                |
| 6    | **AmharicLemmatizer**        | Related-term/synonym lookup             |
| 7    | **AmharicStemmer**           | Rule-based affix stripping (real root extraction) |
| 8    | **AmharicNLP**                | Unified pipeline combining 1-7, plus optional `neural=True` for sentiment/NER/embeddings |
| 9    | **AmharicNeuralPipeline**     | Optional: sentiment, NER, embeddings via pretrained transformer models (`pip install amharicNLP[neural]`) |
| 10   | **AmharicTTS / AmharicASR**   | Optional: text-to-speech and speech-to-text via pretrained models (`pip install amharicNLP[speech]`) |
| 11   | **AmharicDatasetPreprocessor** | Optional: cleans/normalizes a text column of a HF `datasets.Dataset` for voice/LLM/sentiment training (`pip install amharicNLP[datasets]`) |
| 12   | **AmharicCorpusProcessor**    | Loads txt/csv/json/jsonl/DataFrame data; validates & dedups (empty, exact/near-duplicate, corrupt text) |
| 13   | **AmharicLLMDatasetValidator** | Validates instruction/prompt-completion/chat LLM training examples; leakage checks |
| 14   | **AmharicAudioDatasetProcessor** | STT/TTS dataset prep: duration filtering, silence trimming, resampling, speaker-aware splitting |
| 15   | **AmharicVoicePipeline**      | Orchestrates Audio → STT → NLP → (your LLM) → TTS → Audio |

---

# 🖥️ **Command Line**

```bash
amharicnlp tokenize --text "ሰላም ለዓለም"
amharicnlp normalize --text "ኣብዚ ሠላም"
amharicnlp segment --text "ዶ/ር አበበ መጣ። እሱ ጥሩ ነው!"
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






