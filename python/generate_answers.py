from tsv_utils import get_questions
from tsv_utils import write_answers_tsv_2
from generate_base_answers import generate_answers
from config import get_config

import argparse
import uuid
import logging as log

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Create annotated Answers')
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    parser.add_argument('quantity', nargs='?', default=2, help='Amount of created Answers')
    parser.add_argument('--use_sample', action='store_true', help='Use SampleAnswer for more context')
    parser.add_argument('--ignore_text_syntax', action='store_true', help='Ignore spelling or punctuation mistakes for the evaluation')

    # TODO: increase default (2 is only for testing purposes)
    return parser.parse_args()

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)
    log.basicConfig(filename='generate.log', filemode='w')

    args = setup_args()
    config = get_config()
    questions = get_questions(config["data_path"] + config["questions"], args.use_sample)

    for question in questions:
        parameter = (args.api_key, args.quantity, args.ignore_text_syntax) + tuple(question)
        for answer in generate_answers(*parameter):
            row = (question[-1], answer['answer'], str(uuid.uuid4()), answer['rating1'], answer['rating2'])
            write_answers_tsv_2(config["data_path"] + config["answers"], [row], True)
    