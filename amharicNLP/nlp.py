"""
Unified, spaCy-style entry point for amharicNLP.

    from amharicNLP import AmharicNLP

    nlp = AmharicNLP()                 # rule-based only, no extra deps
    doc = nlp("በጣም ጥሩ ፊልም ነበር።")
    doc.tokens                         # ['በጣም', 'ጥሩ', 'ፊልም', 'ነበር', '።']
    doc.stems                          # rule-based affix-stripped stems
    doc.pos_tags                       # [('በጣም','ADV'), ('ጥሩ','NOUN'), ('ፊልም','NOUN'), ('ነበር','AUX'), ('።','PUNCT')]

    nlp = AmharicNLP(neural=True)      # needs: pip install amharicNLP[neural]
    doc = nlp("በጣም ጥሩ ፊልም ነበር።")
    doc.sentiment                      # {'label': 'positive', 'score': 0.97}
    doc.entities                       # [{'entity_group': 'ORG', ...}, ...]
    doc.embedding                      # list[float]

This mirrors the shape of an NLTK/spaCy pipeline (tokenize -> normalize ->
tag/parse) without requiring the heavy dependencies unless you actually
ask for the neural features -- a plain `pip install amharicNLP` stays
lightweight, and `doc.sentiment`/`doc.entities`/`doc.embedding` are lazy:
nothing is loaded into memory until you touch that attribute.
"""

from typing import List, Optional

from .resources import (
    AmharicCleaner,
    AmharicNormalizer,
    AmharicWordTokenizer,
    AmharicSentenceSegmenter,
    AmharicStemmer,
    AmharicStopwordProcessor,
    AmharicPOSTagger,
    AmharicSentimentAnalyzer,
)


class Doc:
    """The result of running AmharicNLP on a piece of text.

    Rule-based fields (tokens, stems, pos_tags, cleaned_text) are computed
    eagerly at construction time -- they're cheap. Neural fields
    (sentiment, entities, embedding) are computed lazily on first access
    and cached, and only work if the Doc was built by a pipeline with
    neural=True.
    """

    __slots__ = (
        "text", "cleaned_text", "tokens", "sentences", "stems", "pos_tags", "sentiment_lexicon",
        "_neural", "_sentiment", "_entities", "_embedding", "_subword_tokens",
    )

    def __init__(self, text: str, cleaned_text: str, tokens: List[str],
                 stems: List[str], pos_tags, sentiment_lexicon, neural=None,
                 sentences: Optional[List[str]] = None):
        self.text = text
        self.cleaned_text = cleaned_text
        self.tokens = tokens
        self.sentences = sentences or []  # sentence-segmented cleaned_text, see AmharicSentenceSegmenter
        self.stems = stems
        self.pos_tags = pos_tags  # list[(word, tag)] -- see AmharicPOSTagger
        self.sentiment_lexicon = sentiment_lexicon  # dict, see AmharicSentimentAnalyzer -- rule-based, always available
        self._neural = neural  # AmharicNeuralPipeline instance, or None
        self._sentiment = None
        self._entities = None
        self._embedding = None
        self._subword_tokens = None

    def _require_neural(self, attr_name: str):
        if self._neural is None:
            raise RuntimeError(
                f"doc.{attr_name} needs the neural pipeline. Build the Doc "
                f"with AmharicNLP(neural=True) (needs: pip install "
                f"amharicNLP[neural])."
            )

    @property
    def sentiment(self):
        self._require_neural("sentiment")
        if self._sentiment is None:
            self._sentiment = self._neural.sentiment(self.text)
        return self._sentiment

    @property
    def entities(self):
        self._require_neural("entities")
        if self._entities is None:
            self._entities = self._neural.entities(self.text)
        return self._entities

    @property
    def embedding(self):
        self._require_neural("embedding")
        if self._embedding is None:
            self._embedding = self._neural.embed(self.text)
        return self._embedding

    @property
    def subword_tokens(self):
        """BPE subword tokens (for LLM/embedding-model input), via the
        real sefineh-ai/Amharic-Tokenizer package -- lazy, computed and
        cached on first access. Distinct from `.tokens` (whole words)."""
        if self._subword_tokens is None:
            from .subword import AmharicSubwordTokenizer
            self._subword_tokens = AmharicSubwordTokenizer().tokenize(self.text)
        return self._subword_tokens

    def __repr__(self):
        return f"Doc(tokens={self.tokens!r})"

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)


class AmharicNLP:
    """The main pipeline object. One instance, reused across calls, so
    models (rule-based and, if enabled, neural) are only built/loaded once.
    """

    def __init__(self, neural: bool = False, remove_stopwords: bool = False):
        self.cleaner = AmharicCleaner()
        self.normalizer = AmharicNormalizer()
        self.tokenizer = AmharicWordTokenizer()
        self.sentence_segmenter = AmharicSentenceSegmenter()
        self.stemmer = AmharicStemmer()
        self.stopwords = AmharicStopwordProcessor()
        self.pos_tagger = AmharicPOSTagger()
        self.sentiment_analyzer = AmharicSentimentAnalyzer()
        self.remove_stopwords = remove_stopwords

        self._neural = None
        if neural:
            from .neural import AmharicNeuralPipeline
            self._neural = AmharicNeuralPipeline()

    def __call__(self, text: str) -> Doc:
        cleaned = self.cleaner.remove_noise(self.cleaner.remove_html(text))
        normalized = self.normalizer.normalize_text(cleaned)
        tokens = self.tokenizer.tokenize(normalized)
        sentences = self.sentence_segmenter.split(normalized)

        if self.remove_stopwords:
            tokens_for_stem = [t for t in tokens if t not in self.stopwords.stop_words]
        else:
            tokens_for_stem = tokens

        stems = [self.stemmer.stemaize(t) for t in tokens_for_stem]
        # POS tagging and lexicon sentiment run on the full (unfiltered)
        # token list -- both need function words/negators present, so
        # they ignore remove_stopwords.
        pos_tags = self.pos_tagger.tag(tokens)
        sentiment_lexicon = self.sentiment_analyzer.analyze(tokens)

        return Doc(text=text, cleaned_text=normalized, tokens=tokens,
                   sentences=sentences, stems=stems, pos_tags=pos_tags,
                   sentiment_lexicon=sentiment_lexicon, neural=self._neural)
