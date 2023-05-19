from tsv_utils import get_questions
from tsv_utils import get_answers_to_rate_per_question
from tsv_utils import write_rated_answers_tsv
from bert_utils import rate_answer
from config import get_config

import logging as log


if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)
    config = get_config()

    questions = get_questions(config["data_path"] + config["questions"], False)
    answers = get_answers_to_rate_per_question(config["data_path"] + config["answers_to_rate"])


    answers_to_rate = []
    for question in questions:
        if question[-1] not in answers:
            log.info(f"No answers to rate for question: {question}")
            continue

        answers_for_question = answers[question[-1]]
        log.debug(f"Training question: {question[0]} answers: {answers_for_question}")
        exit
        # for a single question [{"question_id": 0, "question": "explain foo?", "answers": [{"answer_id": 0, "answer": "a word"}]}]
        answers_to_rate.append({
            "question_id": question[-1],
            "question": question[0],
            "answers": answers_for_question
        })

    rated_answers = rate_answer(config["model_path"], answers_to_rate)

    write_rated_answers_tsv(config["data_path"] + config["rated_answers"], rated_answers, False)