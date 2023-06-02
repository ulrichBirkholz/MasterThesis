from tsv_utils import get_questions
from tsv_utils import get_answers_per_question
from tsv_utils import write_rated_answers_tsv
from bert_utils import rate_answer, AnswersForQuestion
from config import Configuration

import logging as log


if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)
    config = Configuration()

    questions = get_questions(config.get_questions_path(), False)
    answers = get_answers_per_question(config.get_answers_to_rate_path())


    answers_to_rate = []
    for question in questions:
        if question.question_id not in answers:
            log.info(f"No answers to rate for question: {question}")
            continue

        answers_for_question = AnswersForQuestion(question.question, question.question_id, answers[question.question_id])

        log.debug(f"Training question: {question.question} answers: {answers_for_question.answers}")
        answers_to_rate.append(answers_for_question)

    rated_answers = rate_answer(config.get_trained_bert_model_path(), answers_to_rate)

    #TODO: parameterize
    write_rated_answers_tsv(config.get_rated_answers_path('03'), rated_answers, False)