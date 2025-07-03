from amharicNLP.resources.utils import AmharicLanguageDetector

class TestAmharicLanguageDetector:
    def test_is_amharic_text(self):
        detector = AmharicLanguageDetector()
        assert detector.is_amharic_text("ሰላም") is True
        assert detector.is_amharic_text("Hello") is False