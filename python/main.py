from tsv_utils import get_questions
from tsv_utils import write_answers_tsv
from generate_base_answers import generate_answers

import argparse
import uuid
import logging as log

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Create annotated Answers')
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    parser.add_argument('data_path', help='file path and name')
    parser.add_argument('quantity', nargs='?', default=2, help='Amount of created Answers')
    parser.add_argument('--sample', action='store_true', help='Use SampleAnswer for more context')
    parser.add_argument('--ignore_text_syntax', action='store_true', help='Ignore spelling or punctuation mistakes for the evaluation')

    # TODO: increase default (2 is only for testing purposes)
    return parser.parse_args()

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)

    args = setup_args()
    questions = get_questions(args.data_path + '/questions.tsv', args.sample)

    answer_rows = []
    for question in questions:
        parameter = (args.api_key, args.quantity, args.ignore_text_syntax) + tuple(question)
        for answer in generate_answers(*parameter):
            row = (question[-1], answer['answer'], str(uuid.uuid4()), answer['rating'])
            answer_rows.append(row)
            log.debug(f"Got answer: {answer}")

    write_answers_tsv(args.data_path + '/answers.tsv', answer_rows)