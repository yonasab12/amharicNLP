from .resources import *
from .wordnet import *
from .utilities import *
from .nlp import AmharicNLP, Doc
from .subword import AmharicSubwordTokenizer
from .speech import AmharicTTS, AmharicASR
from .dataset_prep import AmharicDatasetPreprocessor
from .corpus import AmharicCorpusProcessor, ValidationReport
from .llm_dataset import (
    AmharicLLMDatasetValidator, LLMValidationReport,
    AmharicLLMDatasetProcessor, LLMPreparedDataset,
)
from .audio_dataset import AmharicAudioDatasetProcessor, SplitResult as AudioSplitResult
from .voice_pipeline import AmharicVoicePipeline, VoiceTurnResult

# --- Bring-Your-Own-Data / Bring-Your-Own-Model layer ----------------------
from .processor import AmharicDatasetProcessor, ProcessingResult, TASKS
from .inspection import inspect_dataset, DatasetReport
from .io_adapters import to_records, to_records_dict, export, EXPORT_FORMATS
from ._optional import OptionalDependencyError, have as has_optional_dependency

# NOTE: amharicNLP.example is intentionally NOT imported here. It used to be
# ("from .example import *"), but example/test.py runs a full demo pipeline
# with print() calls at module scope, so every `import amharicNLP` silently
# ran that demo and dumped output to stdout. Run it explicitly if wanted:
#   python -m amharicNLP.example.test

# ---------------------------------------------------------------------------
# Top-level convenience functions. cli.py imported these
# (`from amharicNLP import tokenize, normalize, analyze_sentiment`) but they
# were never defined anywhere in the package, so the CLI crashed with an
# ImportError on the very first line.
# ---------------------------------------------------------------------------

_tokenizer = AmharicWordTokenizer()
_normalizer = AmharicNormalizer()
_sentence_segmenter = AmharicSentenceSegmenter()
_neural_singleton = None


def tokenize(text: str):
    """Tokenize Amharic text. Thin wrapper around AmharicWordTokenizer."""
    return _tokenizer.tokenize(text)


def split_sentences(text: str):
    """Split Amharic (or mixed Amharic/English) text into sentences.
    Thin wrapper around AmharicSentenceSegmenter.split."""
    return _sentence_segmenter.split(text)


def normalize(text: str) -> str:
    """Normalize Amharic text (character variants + punctuation spacing +
    abbreviation expansion). Thin wrapper around AmharicNormalizer.normalize_text."""
    return _normalizer.normalize_text(text)


def analyze_sentiment(text: str):
    """Sentiment analysis via the optional neural pipeline (a small
    pretrained Amharic transformer model). Needs the 'neural' extra:
        pip install amharicNLP[neural]
    Raises a clear ImportError (not a silent/fake result) if it's missing.
    """
    global _neural_singleton
    if _neural_singleton is None:
        from .neural import AmharicNeuralPipeline
        _neural_singleton = AmharicNeuralPipeline()
    return _neural_singleton.sentiment(text)

