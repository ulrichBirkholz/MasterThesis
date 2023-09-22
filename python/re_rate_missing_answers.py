from config import Configuration
from open_ai_utils_message import annotate_samples
from open_ai_utils_prompt import CHAT_GPT_MODEL
from tsv_utils import get_answers, get_questions, get_key_elements_by_question_id, write_answers_tsv, get_answers_per_question
import argparse
from argparse import Namespace
from config_logger import config_logger
import logging as log


def setup_args() -> Namespace:
    """ Setup of the execution arguments

    Returns:
        Namespace: arguments to be used
    """
    parser = argparse.ArgumentParser(description='Annotate Answers')
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    return parser.parse_args()


if __name__ == "__main__":
    config = Configuration()
    args = setup_args()
    config_logger(log.DEBUG, 're_rate_missing_answers.log')

    turbo_answer_ids = [answer.answer_id for answer in get_answers(config.get_samples_path("turbo_rating_expert_data"))]

    questions = get_questions(config.get_questions_path(), False)
    key_elements_by_question = get_key_elements_by_question_id(config.get_key_elements_path())

    expert_answers = get_answers(config.get_samples_path("experts"))
    expert_answers_per_question = get_answers_per_question

    # Find missing answers
    to_re_rate_per_question = {}
    for e_id in [answer.answer_id for answer in expert_answers]:
        if e_id not in turbo_answer_ids:
            answer = next((answer for answer in expert_answers if answer.answer_id == e_id), None)
            if answer.question_id in to_re_rate_per_question:
                to_re_rate_per_question[answer.question_id].append(answer)
            else:
                to_re_rate_per_question[answer.question_id] = [answer]

    log.debug(f"{sum(len(value) for value in to_re_rate_per_question.values())} Answers are missing")
    # re-rate missing answers and extend the tsv file
    for question in questions:
        if question.question_id in to_re_rate_per_question:
            log.debug(f"About to re-annotate {len(to_re_rate_per_question[question.question_id])} answers for question: {question.question_id}")
            answers = annotate_samples(args.api_key, question, to_re_rate_per_question[question.question_id], key_elements_by_question[question.question_id], CHAT_GPT_MODEL.TURBO)

            write_answers_tsv(config.get_samples_path("turbo_rating_expert_data"), answers, True)
