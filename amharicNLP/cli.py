"""Command-line interface for amharicNLP."""

import argparse

from amharicNLP import tokenize, normalize, analyze_sentiment, split_sentences
from amharicNLP.resources.pos_tagger import AmharicPOSTagger
from amharicNLP.resources.sentiment import AmharicSentimentAnalyzer
from amharicNLP.resources.tokenizer import AmharicWordTokenizer

_pos_tagger = AmharicPOSTagger()
_pos_tokenizer = AmharicWordTokenizer()
_sentiment_tokenizer = AmharicWordTokenizer()
_rule_based_sentiment = AmharicSentimentAnalyzer()


def main():
    parser = argparse.ArgumentParser(description="Amharic NLP Command Line Interface")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Tokenize
    parser_token = subparsers.add_parser("tokenize", help="Tokenize Amharic text")
    parser_token.add_argument("--text", type=str, required=True, help="Amharic text to tokenize")

    # Normalize
    parser_norm = subparsers.add_parser("normalize", help="Normalize Amharic text")
    parser_norm.add_argument("--text", type=str, required=True, help="Amharic text to normalize")

    # Sentence segmentation
    parser_seg = subparsers.add_parser("segment", help="Split Amharic text into sentences")
    parser_seg.add_argument("--text", type=str, required=True, help="Amharic text to segment")

    # Sentiment Analysis. Defaults to the rule-based lexicon analyzer (no
    # extra dependencies, same as the "pos" command below) so this command
    # works out of the box on a plain `pip install amharicNLP`. Pass
    # --neural to use the real fine-tuned transformer model instead, which
    # is more accurate but needs `pip install amharicNLP[neural]`.
    parser_sent = subparsers.add_parser(
        "sentiment",
        help="Sentiment analysis for Amharic text (rule-based by default; see --neural)",
    )
    parser_sent.add_argument("--text", type=str, required=True, help="Amharic text for sentiment analysis")
    parser_sent.add_argument(
        "--neural",
        action="store_true",
        help="Use the neural (transformer) sentiment model instead of the rule-based lexicon. "
             "Needs: pip install amharicNLP[neural]",
    )

    # POS tagging (rule-based, no dependency)
    parser_pos = subparsers.add_parser("pos", help="Rule-based POS tagging for Amharic text")
    parser_pos.add_argument("--text", type=str, required=True, help="Amharic text to tag")

    args = parser.parse_args()

    if args.command == "tokenize":
        tokens = tokenize(args.text)
        print("🔹 Tokens:", tokens)

    elif args.command == "normalize":
        norm_text = normalize(args.text)
        print("🔹 Normalized:", norm_text)

    elif args.command == "segment":
        sentences = split_sentences(args.text)
        print("🔹 Sentences:", sentences)

    elif args.command == "sentiment":
        if args.neural:
            sentiment = analyze_sentiment(args.text)
        else:
            tokens = _sentiment_tokenizer.tokenize(args.text)
            sentiment = _rule_based_sentiment.analyze(tokens)
        print("🔹 Sentiment:", sentiment)

    elif args.command == "pos":
        tagged = _pos_tagger.tag(_pos_tokenizer.tokenize(args.text))
        print("🔹 POS tags:", tagged)


if __name__ == "__main__":
    main()
