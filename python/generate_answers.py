from tsv_utils import get_questions, write_answers_tsv, get_key_elements_by_question_id
from generate_base_answers import generate_answers
from config import Configuration


import argparse
import logging as log

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Create annotated Answers')
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    parser.add_argument('batch_size', default=2, type=int, help='Each of the 600 batches, generates "batch_size" number of answers. With a batch size of 3, this equates to 1,800 answers being produced.')
    parser.add_argument('--use_sample', action='store_true', help='Use SampleAnswer for more context')
    return parser.parse_args()

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)
    log.basicConfig(filename='generate.log', filemode='w')

    args = setup_args()
    config = Configuration()
    questions = get_questions(config.get_questions_path(), args.use_sample)
    key_elements_per_question = get_key_elements_by_question_id(config.get_key_elements_path())

    unrated_answer_path = config.get_ai_unrated_answer_path()
    # We target 4k Answers per Question in total -> batch_size of 7
    for question in questions:
        key_elements = key_elements_per_question[question.question_id]
        # we use yield to iterate over every generated response and save after every performed request
        write_answers_tsv(unrated_answer_path, generate_answers(args.api_key, args.batch_size, question, key_elements), True)
    