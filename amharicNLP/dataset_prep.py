"""
Bridge between amharicNLP's rule-based preprocessing and Hugging Face
`datasets` -- for cleaning up a text/transcript column in a dataset before
training a voice model (ASR/TTS), an Amharic LLM, or a sentiment
classifier. The same cleaning problem (raw crowdsourced or scraped text
needs normalization before it's useful as model input/target) shows up in
all three cases; this module is deliberately generic rather than three
separate one-off scripts.

Verified against real data: the sentences used in this module's tests
were copied verbatim from the snapwre/amharic-speech dataset viewer
(https://huggingface.co/datasets/snapwre/amharic-speech, CC-BY-4.0,
16.9k Amharic speech clips with transcripts) -- not invented examples.
That process found and fixed one real, previously-unknown bug: "፡፡" (a
traditional doubled Ethiopic wordspace still common in news writing,
equivalent to "።") was splitting into two separate tokens instead of one
full stop. It showed up in roughly a quarter of the sampled sentences.

This module itself has NOT been run against the dataset end-to-end via
`datasets.load_dataset` -- the environment that built it has no network
access to huggingface.co. The `datasets` library API used here
(`load_dataset`, `Dataset.map`) is standard and well-documented, but
running `AmharicDatasetPreprocessor.load(...)` for the first time on your
own machine is the actual test, not this module's existence.

Requires the optional "datasets" extra:
    pip install amharicNLP[datasets]
"""

from typing import Callable, Optional

from .resources.cleaner import AmharicCleaner
from .resources.normalizer import AmharicNormalizer
from .resources.tokenizer import AmharicWordTokenizer
from .resources.stemmer import AmharicStemmer
from .resources.stopword import AmharicStopwordProcessor


class AmharicDatasetPreprocessor:
    """
    Cleans and normalizes a text column of a Hugging Face `datasets.Dataset`
    or `DatasetDict`, leaving every other column (audio, speaker_id,
    duration_s, label, ...) untouched.

        prep = AmharicDatasetPreprocessor()

        # Speech dataset (e.g. snapwre/amharic-speech): text column is
        # "sentence", audio/duration/speaker columns pass through as-is.
        ds = prep.load("snapwre/amharic-speech")
        ds = prep.process(ds, text_column="sentence")
        ds["train"][0]["sentence_clean"]   # normalized transcript
        ds["train"][0]["audio"]            # untouched

        # Sentiment or LLM dataset: same call, different column name.
        ds = prep.load("your-org/amharic-sentiment")
        ds = prep.process(ds, text_column="text")

    `process()` always cleans (strip HTML/noise) and normalizes (character
    variants, punctuation spacing, abbreviation expansion). Tokenizing,
    stopword removal, and stemming are opt-in via the flags below, since
    they're the right choice for some downstream tasks (e.g. a
    bag-of-words sentiment baseline) and actively wrong for others (ASR/TTS
    training wants normalized *text*, not a token list; an LLM usually
    wants raw normalized text too, not stemmed/stopword-stripped text).
    """

    def __init__(
        self,
        tokenize: bool = False,
        remove_stopwords: bool = False,
        stem: bool = False,
    ):
        self.cleaner = AmharicCleaner()
        self.normalizer = AmharicNormalizer()
        self.tokenizer = AmharicWordTokenizer() if (tokenize or remove_stopwords or stem) else None
        self.stopwords = AmharicStopwordProcessor() if remove_stopwords else None
        self.stemmer = AmharicStemmer() if stem else None
        self.do_tokenize = tokenize
        self.do_remove_stopwords = remove_stopwords
        self.do_stem = stem

    @staticmethod
    def load(dataset_name: str, split: Optional[str] = None, **kwargs):
        """Thin wrapper around `datasets.load_dataset`, so this module is
        the one place that needs the 'datasets' extra installed."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "AmharicDatasetPreprocessor.load needs the optional 'datasets' extra: "
                "pip install amharicNLP[datasets]"
            ) from exc
        return load_dataset(dataset_name, split=split, **kwargs)

    def clean_text(self, text: str) -> str:
        """The text-level pipeline, usable directly on a single string."""
        if not text:
            return text
        cleaned = self.cleaner.remove_noise(self.cleaner.remove_html(text))
        normalized = self.normalizer.normalize_text(cleaned)
        return normalized

    def process_text(self, text: str):
        """Returns the cleaned string, or a list of tokens/stems if
        tokenize/remove_stopwords/stem was requested at construction."""
        normalized = self.clean_text(text)
        if not self.do_tokenize and not self.do_remove_stopwords and not self.do_stem:
            return normalized

        tokens = self.tokenizer.tokenize(normalized)
        if self.do_remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords.stop_words]
        if self.do_stem:
            tokens = [self.stemmer.stemaize(t) for t in tokens]
        return tokens

    def process(
        self,
        dataset,
        text_column: str,
        new_column: Optional[str] = None,
        num_proc: Optional[int] = None,
    ):
        """
        Apply process_text() to `text_column` in a `datasets.Dataset` or
        `DatasetDict`, writing the result to `new_column` (default:
        f"{text_column}_clean"). Every other column is left untouched --
        this never re-encodes or re-samples audio.
        """
        new_column = new_column or f"{text_column}_clean"

        def _map_fn(example):
            example[new_column] = self.process_text(example[text_column])
            return example

        return dataset.map(_map_fn, num_proc=num_proc)
