from tsv_utils import get_questions, write_answers_tsv, get_key_elements_by_question_id
from open_ai_utils_davinci import generate_samples as generate_samples_davinci
from open_ai_utils_message import generate_samples as generate_samples_message, CHAT_GPT_MODEL
from config import Configuration

import argparse
from argparse import Namespace
import logging as log
from config_logger import config_logger


def setup_args() -> Namespace:
    """ Setup of the execution arguments

    Returns:
        Namespace: arguments to be used
    """
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

    unrated_samples_davinci_path = config.get_unrated_samples_path("davinci")
    unrated_samples_gpt4_path = config.get_unrated_samples_path("gpt4")

    for question in questions:
        key_elements = key_elements_per_question[question.question_id]
        # we use yield to iterate over every generated response and save after every performed request
        # producing answers with gpt-3.5-turbo did not work properly because of its refusal to create incorrect answers
        write_answers_tsv(unrated_samples_davinci_path, generate_samples_davinci(args.api_key, question, key_elements), True)
        write_answers_tsv(unrated_samples_gpt4_path, generate_samples_message(args.api_key, question, key_elements, CHAT_GPT_MODEL.GPT4), True)
    
    