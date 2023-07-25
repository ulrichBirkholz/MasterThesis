from tsv_utils import get_questions, write_answers_tsv, get_answers_per_question, get_key_elements_by_question_id
from generate_base_answers import rate_answers
from config import Configuration

import argparse
import logging as log
from config_logger import config_logger

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Create annotated Answers')
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    parser.add_argument('score_type', default=2, type=int, help='Wether the annotation is for score_1 or score_2')
    # 2 produces stable results, 5 is unstable, so some responses are unparsable 10 and higher was unusable
    parser.add_argument('chunk_size', default=2, type=int, help='Maximal amount of simultaneously annotated Answers')
    return parser.parse_args()

def _chunk(answers, size:int):
    for i in range(0, len(answers), size):
        yield answers[i:i + size]

if __name__ == "__main__":
    config_logger(log.DEBUG, 'annotate.log')

    args = setup_args()
    config = Configuration()
    questions = get_questions(config.get_questions_path(), args.use_sample)
    key_elements_per_question = get_key_elements_by_question_id(config.get_key_elements_path())

    # map could be used but makes it less readable
    answer_path = config.get_ai_answers_path()
    unrated_answer_path = config.get_ai_unrated_answer_path()
    unrated_answers = get_answers_per_question(config.get_ai_unrated_answer_path())

    for question_id, answers in unrated_answers.items():
        key_elements = key_elements_per_question[question_id]
        question = next(filter(lambda question: question.question_id == question_id, questions))
        for chunk in _chunk(answers, args.chunk_size):
            write_answers_tsv(answer_path, rate_answers(args.api_key, question, chunk, key_elements, args.score_type), True)

            
    