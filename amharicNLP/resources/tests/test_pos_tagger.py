from amharicNLP.resources.pos_tagger import AmharicPOSTagger


class TestAmharicPOSTagger:
    def setup_method(self):
        self.tagger = AmharicPOSTagger()

    def test_pronoun(self):
        assert self.tagger.tag_word("እኔ") == "PRON"

    def test_punctuation(self):
        assert self.tagger.tag_word("።") == "PUNCT"

    def test_digit(self):
        assert self.tagger.tag_word("123") == "NUM"

    def test_number_word(self):
        assert self.tagger.tag_word("ሁለት") == "NUM"

    def test_coordinating_conjunction(self):
        assert self.tagger.tag_word("እና") == "CCONJ"

    def test_subordinating_conjunction(self):
        assert self.tagger.tag_word("ስለዚህ") == "SCONJ"

    def test_auxiliary_copula(self):
        assert self.tagger.tag_word("ነው") == "AUX"

    def test_adposition(self):
        assert self.tagger.tag_word("ውስጥ") == "ADP"

    def test_negated_verb(self):
        assert self.tagger.tag_word("አልሄደም") == "VERB"

    def test_negated_verb_epenthetic_variant(self):
        # The አል...ም negation circumfix commonly surfaces as አላ...ም before
        # certain root-initial consonant clusters (didn't read / didn't win).
        assert self.tagger.tag_word("አላነበብኩም") == "VERB"
        assert self.tagger.tag_word("አላሸነፈም") == "VERB"

    def test_perfective_verb_with_no_prefix(self):
        # Amharic perfective (past) conjugation is suffix-only, no subject
        # prefix -- must not require has_prefix AND has_suffix.
        assert self.tagger.tag_word("አበሰለች") == "VERB"  # she cooked

    def test_plural_noun_not_misread_as_verb(self):
        # Regression test: -ኞች/-ዎች (plural) and -ውን (definite object) share
        # their last letter with verb person-agreement suffixes (ች/ን). A
        # bare endswith(ች)/endswith(ን) check on the perfective-suffix path
        # misfires on very common nouns; these must stay NOUN.
        assert self.tagger.tag_word("ሰራተኞች") == "NOUN"  # workers
        assert self.tagger.tag_word("ተማሪዎች") == "NOUN"  # students
        assert self.tagger.tag_word("ጨዋታውን") == "NOUN"  # the game[obj]

    def test_bare_verb_root(self):
        assert self.tagger.tag_word("ሄደ") == "VERB"

    def test_adjective_suffix(self):
        assert self.tagger.tag_word("ኢትዮጵያዊ") == "ADJ"

    def test_known_proper_noun(self):
        assert self.tagger.tag_word("ኢትዮጵያ") == "PROPN"

    def test_unknown_word_defaults_to_noun(self):
        assert self.tagger.tag_word("ኮምፒውተር") == "NOUN"

    def test_tag_sentence(self):
        tokens = ["እኔ", "ኢትዮጵያዊ", "ነኝ", "።"]
        result = self.tagger.tag(tokens)
        assert result == [
            ("እኔ", "PRON"),
            ("ኢትዮጵያዊ", "ADJ"),
            ("ነኝ", "AUX"),
            ("።", "PUNCT"),
        ]
