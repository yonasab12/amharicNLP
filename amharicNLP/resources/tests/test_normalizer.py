from amharicNLP .resources.normalizer import AmharicNormalizer

class TestAmharicNormalizer:
    def test_normalize(self):
        normalizer = AmharicNormalizer()
        assert normalizer.normalize("ኣብዚ") == "አብዚ"  # Example normalization