# cli.py

import argparse
from amharicNLP import tokenize, normalize, analyze_sentiment
from amharicNLP.resources.pos_tagger import AmharicPOSTagger
from amharicNLP.resources.tokenizer import AmharicWordTokenizer

_pos_tagger = AmharicPOSTagger()
_pos_tokenizer = AmharicWordTokenizer()

def main():
    parser = argparse.ArgumentParser(description="Amharic NLP Command Line Interface")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Tokenize
    parser_token = subparsers.add_parser("tokenize", help="Tokenize Amharic text")
    parser_token.add_argument("--text", type=str, required=True, help="Amharic text to tokenize")

    # Normalize
    parser_norm = subparsers.add_parser("normalize", help="Normalize Amharic text")
    parser_norm.add_argument("--text", type=str, required=True, help="Amharic text to normalize")

    # Sentiment Analysis
    parser_sent = subparsers.add_parser("sentiment", help="Sentiment analysis for Amharic text")
    parser_sent.add_argument("--text", type=str, required=True, help="Amharic text for sentiment analysis")

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

    elif args.command == "sentiment":
        sentiment = analyze_sentiment(args.text)
        print("🔹 Sentiment:", sentiment)

    elif args.command == "pos":
        tagged = _pos_tagger.tag(_pos_tokenizer.tokenize(args.text))
        print("🔹 POS tags:", tagged)

if __name__ == "__main__":
    main()
