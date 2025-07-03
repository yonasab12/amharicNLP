import pytest
from amharicNLP.resources import AmharicCleaner, AmharicNormalizer

@pytest.fixture
def sample_amharic_text():
    return "ይህ የሙከራ ጽሑፍ ነው።"

@pytest.fixture
def cleaner():
    return AmharicCleaner()

@pytest.fixture
def normalizer():
    return AmharicNormalizer()