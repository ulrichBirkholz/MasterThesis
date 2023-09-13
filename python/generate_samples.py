from tsv_utils import get_questions, write_answers_tsv, get_key_elements_by_question_id
from open_ai_utils import generate_samples
from config import Configuration


import argparse
import logging as log
from config_logger import config_logger

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Create annotated Answers')
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    parser.add_argument('--use_sample', action='store_true', help='Use SampleAnswer for more context')
    return parser.parse_args()

if __name__ == "__main__":
    config_logger(log.DEBUG, "generate.log")

    args = setup_args()
    config = Configuration()
    questions = get_questions(config.get_questions_path(), args.use_sample)
    key_elements_per_question = get_key_elements_by_question_id(config.get_key_elements_path())

    unrated_samples_path = config.get_unrated_samples_path()
    # We target 4k Answers per Question in total -> batch_size of 7
    for question in questions:
        key_elements = key_elements_per_question[question.question_id]
        # we use yield to iterate over every generated response and save after every performed request
        write_answers_tsv(unrated_samples_path, generate_samples(args.api_key, question, key_elements), True)
    