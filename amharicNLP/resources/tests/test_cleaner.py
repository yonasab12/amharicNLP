import pytest
from amharicNLP .resources.cleaner import AmharicCleaner

class TestAmharicCleaner:
    @pytest.fixture
    def cleaner(self):
        return AmharicCleaner()
    
    def test_remove_html(self, cleaner):
        assert cleaner.remove_html("<p>Hello</p>") == "Hello"
    
    def test_remove_noise(self, cleaner):
        text = "ሰላም! 😊 Hello 123"
        assert cleaner.remove_noise(text) == "ሰላም! 123"