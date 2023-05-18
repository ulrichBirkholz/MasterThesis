from transformers import BertTokenizer
import argparse


def get_tokens(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    return tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Count the tokens of a given text')
    parser.add_argument('text', help='The text to count')
    args = parser.parse_args()

    token = get_tokens(args.text)
    print(f"Token: {token}")
    print(f"Token count: {len(token)}")
