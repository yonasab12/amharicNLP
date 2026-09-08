"""
Optional neural layer for amharicNLP.

The rule-based tools in amharicNLP.resources (tokenizer, stemmer,
normalizer, cleaner) have zero third-party dependencies and are fast, but
they can't do things that need learned patterns: sentiment, named-entity
recognition, or semantic similarity. Rather than training new models from
scratch (expensive, and unnecessary), this module wraps small, free,
Amharic-specific transformer models that are already pretrained and
published on the Hugging Face Hub:

    - rasyosef/bert-medium-amharic-finetuned-sentiment  (~40M params)
    - rasyosef/bert-medium-amharic-finetuned-ner         (~40M params)
    - rasyosef/bert-medium-amharic                       (~40M params, embeddings)

These were pretrained from scratch on ~290M tokens of Amharic text and,
per their model cards, outperform 279M-param multilingual XLM-R on
Amharic sentiment/NER despite being ~7x smaller. They run fine on CPU for
inference -- no GPU or large compute budget is needed to use them. Fine
alternatives if you want to swap models: EthioNLP/Amharic-LLAMA-all-data
(generation), other entries in the rasyosef "Amharic BERT and RoBERTa"
collection (tiny/mini/small/base sizes) for a size/accuracy trade-off.

This module requires the optional "neural" extra:
    pip install amharicNLP[neural]
It is never imported by amharicNLP/__init__.py, so a plain
`pip install amharicNLP` stays dependency-light as advertised.
"""

from typing import List, Dict, Optional


class AmharicNeuralPipeline:
    """Lazy-loading wrapper around pretrained Amharic transformer models.

    Nothing is downloaded or loaded into memory until you call a method
    that needs it, and each model is loaded at most once per instance.
    """

    SENTIMENT_MODEL = "rasyosef/bert-medium-amharic-finetuned-sentiment"
    NER_MODEL = "rasyosef/bert-medium-amharic-finetuned-ner"
    EMBEDDING_MODEL = "rasyosef/bert-medium-amharic"

    def __init__(
        self,
        sentiment_model: Optional[str] = None,
        ner_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.sentiment_model = sentiment_model or self.SENTIMENT_MODEL
        self.ner_model = ner_model or self.NER_MODEL
        self.embedding_model = embedding_model or self.EMBEDDING_MODEL
        self.device = device

        self._sentiment_pipe = None
        self._ner_pipe = None
        self._embed_tokenizer = None
        self._embed_model = None

    @staticmethod
    def _require_transformers():
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "amharicNLP.neural needs the optional 'neural' extra: "
                "pip install amharicNLP[neural]"
            ) from exc

    def sentiment(self, text: str) -> Dict[str, object]:
        """Return {'label': ..., 'score': ...} for a piece of Amharic text."""
        self._require_transformers()
        if self._sentiment_pipe is None:
            from transformers import pipeline
            self._sentiment_pipe = pipeline(
                "text-classification", model=self.sentiment_model, device=self.device
            )
        return self._sentiment_pipe(text)[0]

    def entities(self, text: str) -> List[Dict[str, object]]:
        """Return a list of {'entity_group','word','score','start','end'} spans."""
        self._require_transformers()
        if self._ner_pipe is None:
            from transformers import pipeline
            self._ner_pipe = pipeline(
                "token-classification",
                model=self.ner_model,
                aggregation_strategy="simple",
                device=self.device,
            )
        return self._ner_pipe(text)

    def embed(self, text: str) -> List[float]:
        """Return a mean-pooled sentence embedding as a plain list of floats."""
        self._require_transformers()
        import torch

        if self._embed_model is None:
            from transformers import AutoTokenizer, AutoModel
            self._embed_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
            self._embed_model = AutoModel.from_pretrained(self.embedding_model)
            self._embed_model.eval()
            if self.device:
                self._embed_model.to(self.device)

        with torch.no_grad():
            inputs = self._embed_tokenizer(text, return_tensors="pt", truncation=True)
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._embed_model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            pooled = summed / counts
        return pooled.squeeze(0).tolist()

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts' embeddings."""
        import math

        a, b = self.embed(text_a), self.embed(text_b)
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
