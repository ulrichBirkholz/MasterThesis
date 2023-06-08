from tsv_utils import get_questions
from tsv_utils import write_answers_tsv_2
from generate_base_answers import generate_answers
from config import Configuration


import argparse
import uuid
import logging as log

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Create annotated Answers')
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    
    # 2 produces stable results, 5 is unstable, so some responses are unparsable 10 and higher was unusable
    parser.add_argument('quantity', default=2, help='Amount of created Answers')
    parser.add_argument('--use_sample', action='store_true', help='Use SampleAnswer for more context')
    parser.add_argument('--ignore_text_syntax', action='store_true', help='Ignore spelling or punctuation mistakes for the evaluation')
    return parser.parse_args()

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)
    log.basicConfig(filename='generate.log', filemode='w')

    args = setup_args()
    config = Configuration()
    questions = get_questions(config.get_questions_path(), args.use_sample)

    # map could be used but makes it less readable
    answer_path = config.get_ai_answers_path()
    for question in questions:
        for answer in generate_answers(args.api_key, args.quantity, args.ignore_text_syntax, question):
            write_answers_tsv_2(answer_path, [answer], True)
    