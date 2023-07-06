from tsv_utils import get_questions, write_answers_tsv, get_answers, get_key_elements_by_question_id, Answer
from generate_base_answers import rate_answers
from config import Configuration

import logging as log

def _chunk(answers, size:int):
    for i in range(0, len(answers), size):
        yield answers[i:i + size]

def _invalid_rating(answer:Answer):
    score = int(answer.score_2)
    return score < 0 or score >= 4

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)
    log.basicConfig(filename='generate.log', filemode='w')

    config = Configuration()

    answers_1 = get_answers(config.get_ai_answers_path("01"))
    answers_2 = get_answers(config.get_ai_answers_path("02"))

    for answer_1 in answers_1:
        for answer_2 in answers_2:
            if answer_1.answer_id == answer_2.answer_id:
                answer_1.score_1 = answer_2.score_2

    write_answers_tsv(config.get_ai_answers_path(), [answers_1], True)