from amharicNLP.resources.normalizer import AmharicNormalizer

class TestAmharicNormalizer:
    def test_normalize(self):
        normalizer = AmharicNormalizer()
        assert normalizer.normalize("ኣብዚ") == "አብዚ"  # Example normalization

    def test_doubled_wordspace_becomes_full_stop(self):
        # Regression test: "፡፡" (traditional doubled Ethiopic wordspace,
        # still common in news writing) is equivalent to "።" but was being
        # split into two separate spaced "፡" characters instead. Found by
        # testing against real transcripts from the snapwre/amharic-speech
        # dataset, where it appeared in roughly a quarter of sampled
        # sentences.
        normalizer = AmharicNormalizer()
        assert normalizer.normalize_text("ብለዋል፡፡") == "ብለዋል ።"

    def test_hoy_sadis_variant_normalizes_consistently(self):
        # Regression test: 'ኅ' (6th-order ኀ-series) used to appear twice in
        # AMHARIC_NORMALIZATION_MAP with two different targets ('ሀ' then
        # 'ህ') -- a duplicate dict key, so Python silently kept only the
        # last one. Pin down the (correct) behavior: 'ኅ' normalizes the
        # same way its exact 6th-order counterpart 'ሕ' does.
        normalizer = AmharicNormalizer()
        assert normalizer.normalize("ኅ") == normalizer.normalize("ሕ") == "ህ"

    def test_abbreviation_expansion_has_no_corrupted_characters(self):
        # Regression test: the ፍ/ስላሴ abbreviation expanded to text
        # containing a literal Unicode replacement character (mojibake).
        normalizer = AmharicNormalizer()
        expanded = normalizer.expand_abbreviations("ፍ/ስላሴ")
        assert "\ufffd" not in expanded
        assert expanded == "ፍቅረ ስላሴ"

    def test_spelling_correction_does_not_corrupt_substring_matches(self):
        # Regression test: fix_common_spelling used a blind text.replace(),
        # which also matched 'wrong' inside unrelated longer words (e.g.
        # correcting 'አበ' -> 'አበራ' also mangled 'አበባ' "flower" into
        # 'አበራባ'). Now word-boundary-aware.
        normalizer = AmharicNormalizer()
        assert normalizer.fix_common_spelling("አበባ ውብ ናት") == "አበባ ውብ ናት"
        assert normalizer.fix_common_spelling("አበ መጣ") == "አበራ መጣ"


class TestSocialMediaAndNoiseHandling:
    def setup_method(self):
        self.n = AmharicNormalizer()

    def test_normalize_urls_placeholder_default(self):
        assert self.n.normalize_urls("ይመልከቱ https://example.com/path እባክዎ") == \
            "ይመልከቱ <URL> እባክዎ"

    def test_normalize_urls_www_without_scheme(self):
        assert self.n.normalize_urls("ይመልከቱ www.example.com እባክዎ") == \
            "ይመልከቱ <URL> እባክዎ"

    def test_normalize_urls_remove(self):
        result = self.n.normalize_urls("ይመልከቱ https://example.com እባክዎ", mode="remove")
        assert "https://" not in result
        assert "example.com" not in result

    def test_normalize_urls_keep_is_noop(self):
        text = "ይመልከቱ https://example.com እባክዎ"
        assert self.n.normalize_urls(text, mode="keep") == text

    def test_normalize_urls_invalid_mode_raises(self):
        import pytest
        with pytest.raises(ValueError):
            self.n.normalize_urls("x", mode="bogus")

    def test_normalize_emails_placeholder(self):
        assert self.n.normalize_emails("ላክልኝ test.user@example.com እባክህ") == \
            "ላክልኝ <EMAIL> እባክህ"

    def test_normalize_emails_remove(self):
        result = self.n.normalize_emails("ላክልኝ test@example.com እባክህ", mode="remove")
        assert "@" not in result

    def test_normalize_hashtags_strip_symbol_keeps_amharic_word(self):
        assert self.n.normalize_hashtags("እወዳለሁ #ኢትዮጵያ እና #Ethiopia") == \
            "እወዳለሁ ኢትዮጵያ እና Ethiopia"

    def test_normalize_hashtags_remove(self):
        assert self.n.normalize_hashtags("እወዳለሁ #ኢትዮጵያ ሀገር", mode="remove") == \
            "እወዳለሁ  ሀገር"

    def test_normalize_mentions_placeholder_default(self):
        assert self.n.normalize_mentions("ሰላም @user1 እንዴት ነህ") == \
            "ሰላም <USER> እንዴት ነህ"

    def test_normalize_mentions_strip_symbol(self):
        assert self.n.normalize_mentions("ሰላም @user1 እንዴት", mode="strip_symbol") == \
            "ሰላም user1 እንዴት"

    def test_hashtag_not_confused_with_mention(self):
        # '#' and '@' regexes shouldn't cross-match each other's marker.
        text = "#ታግ እና @ተጠቃሚ"
        hashtag_only = self.n.normalize_hashtags(text, mode="strip_symbol")
        assert "@ተጠቃሚ" in hashtag_only

    def test_collapse_repeated_characters_default_max_two(self):
        assert self.n.collapse_repeated_characters("ደሓናናናና ነኝ") == "ደሓናና ነኝ"
        assert self.n.collapse_repeated_characters("soooo good") == "soo good"

    def test_collapse_repeated_characters_custom_max(self):
        assert self.n.collapse_repeated_characters("ደሓናናናና", max_repeat=1) == "ደሓና"

    def test_collapse_repeated_characters_leaves_short_runs_alone(self):
        assert self.n.collapse_repeated_characters("ጥሩ ነው") == "ጥሩ ነው"

    def test_collapse_repeated_characters_invalid_max_raises(self):
        import pytest
        with pytest.raises(ValueError):
            self.n.collapse_repeated_characters("x", max_repeat=0)

    def test_collapse_repeated_punctuation_latin(self):
        assert self.n.collapse_repeated_punctuation("እውነት ነው???") == "እውነት ነው?"
        assert self.n.collapse_repeated_punctuation("ጠብቅ....") == "ጠብቅ."

    def test_collapse_repeated_punctuation_amharic(self):
        assert self.n.collapse_repeated_punctuation("እሺ፣፣፣ ደህና") == "እሺ፣ ደህና"

    def test_normalize_text_opts_in_to_social_media_handling(self):
        # url_mode/email_mode/hashtag_mode/mention_mode are None (skip) by
        # default -- normalize_text should not touch a plain URL unless
        # explicitly asked to.
        n = self.n
        text = "ይመልከቱ https://example.com"
        assert "<URL>" not in n.normalize_text(text, expand_abbr=False)
        assert "<URL>" in n.normalize_text(text, url_mode="placeholder", expand_abbr=False)

    def test_normalize_text_collapses_repeats_when_requested(self):
        result = self.n.normalize_text(
            "ደሓናናናና!!! ነኝ", collapse_repeated_chars=True, collapse_repeated_punct=True,
            expand_abbr=False,
        )
        assert "ናናናና" not in result
        assert "!!!" not in result