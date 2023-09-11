from tsv_utils import get_questions, write_answers_tsv, get_answers_per_question, get_key_elements_by_question_id
from open_ai_utils import annotate_samples
from config import Configuration

import argparse
import logging as log
from config_logger import config_logger

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Create annotated Answers')
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    # 2 produces stable results, 5 is unstable, so some responses are unparsable 10 and higher was unusable
    parser.add_argument('chunk_size', default=2, type=int, help='Maximal amount of simultaneously annotated Answers')
    return parser.parse_args()


def _chunk(answers, size:int):
    for i in range(0, len(answers), size):
        yield answers[i:i + size]


def _process_answers(answers_per_question, questions, key_elements_per_question, answers_path, api_key, chunk_size):
    for question_id, answers in answers_per_question.items():
        key_elements = key_elements_per_question[question_id]
        
        # find question by id
        question = next(filter(lambda question: question.question_id == question_id, questions), None)
        if question:
            for chunk in _chunk(answers, chunk_size):
                write_answers_tsv(answers_path, annotate_samples(api_key, question, chunk, key_elements), True)
        else:
            log.error(f"no matching question found for Id: {question_id}")


if __name__ == "__main__":
    config_logger(log.DEBUG, 'annotate.log')

    args = setup_args()
    config = Configuration()
    questions = get_questions(config.get_questions_path(), False)
    key_elements_per_question = get_key_elements_by_question_id(config.get_key_elements_path())

    ai_rated_man_answers_path = config.get_ai_rated_man_answers_path()
    ai_rated_answer_path = config.get_ai_answers_path()

    unrated_ai_answers_per_question = get_answers_per_question(config.get_ai_unrated_answer_path())
    man_answers_per_question = get_answers_per_question(config.get_man_answers_path())

    _process_answers(unrated_ai_answers_per_question, questions, key_elements_per_question, ai_rated_answer_path, args.api_key, args.chunk_size)
    _process_answers(man_answers_per_question, questions, key_elements_per_question, ai_rated_man_answers_path, args.api_key, args.chunk_size)
