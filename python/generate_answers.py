from tsv_utils import get_questions, write_answers_tsv, get_key_elements_by_question_id
from generate_base_answers import generate_answers
from config import Configuration


import argparse
import logging as log

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Create annotated Answers')
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    
    # 2 produces stable results, 5 is unstable, so some responses are unparsable 10 and higher was unusable
    parser.add_argument('quantity', default=2, type=int, help='Amount of created Answers per category')
    parser.add_argument('--use_sample', action='store_true', help='Use SampleAnswer for more context')
    parser.add_argument('--ignore_text_syntax', action='store_true', help='Ignore spelling or punctuation mistakes for the evaluation')
    return parser.parse_args()

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)
    log.basicConfig(filename='generate.log', filemode='w')

    args = setup_args()
    config = Configuration()
    questions = get_questions(config.get_questions_path(), args.use_sample)
    key_elements_per_question = get_key_elements_by_question_id(config.get_key_elements_path())

    unrated_answer_path = config.get_ai_unrated_answer_path()
    # We target 4k Answers per Question in total
    for question in questions:
        key_elements = key_elements_per_question[question.question_id]
        # we use yield to iterate over every generated response and save after every performed request
        write_answers_tsv(unrated_answer_path, generate_answers(args.api_key, args.quantity, args.ignore_text_syntax, question, key_elements), True)
    