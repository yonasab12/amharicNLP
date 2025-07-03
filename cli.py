# cli.py

import argparse
from amharicNLP import tokenize, normalize, analyze_sentiment

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

if __name__ == "__main__":
    main()
