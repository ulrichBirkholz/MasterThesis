from tsv_utils import get_questions
from tsv_utils import get_answers_per_question
from tsv_utils import write_rated_answers_tsv
from bert_utils import rate_answer, AnswersForQuestion
from config import Configuration
from tsv_utils import Question
from typing import Dict
import random

import logging as log

def _rate_answers(question: Question, answers_for_question:Dict, config:Configuration, model_descriptor:str, answer_descriptor:str):
    if question.question_id not in answers_for_question:
            log.info(f"No answers to rate for question: {question}")
            return
    answers = answers_for_question[question.question_id]

    # we rate max. 1000 answers
    if len(answers) > 1000:
          answers = random.sample(answers, 1000)
 
    for batch_size in config.get_batch_sizes():
        for id in batch_size.ids:
            log.debug(f"Rate Answers for question: {question.question_id} with batch size: {batch_size.size}, id: {id}, model_descriptor: {model_descriptor}")
            rated_answers = rate_answer(config.get_trained_bert_model_path(question.question, batch_size.size, id, model_descriptor), AnswersForQuestion(question.question, question.question_id, answers))
            write_rated_answers_tsv(config.get_rated_answers_path(model_descriptor, answer_descriptor, batch_size.size, id), rated_answers, False)

if __name__ == "__main__":
    log.basicConfig(level=log.DEBUG)
    config = Configuration()

    questions = get_questions(config.get_questions_path(), False)
    ai_answers_rating = get_answers_per_question(config.get_ai_answers_to_rate_path())
    man_answers_rating = get_answers_per_question(config.get_man_answers_to_rate_path())
    ai_answers_training = get_answers_per_question(config.get_ai_answers_for_training_path())
    man_answers_training = get_answers_per_question(config.get_man_answers_for_training_path())

    for question in questions:
        _rate_answers(question, ai_answers_training, config, 'ai', 'ai-training')
        _rate_answers(question, ai_answers_rating, config, 'ai', 'ai-rating')
        _rate_answers(question, man_answers_rating, config, 'ai', 'man-rating')
        _rate_answers(question, man_answers_training, config, 'man', 'man-training')
        _rate_answers(question, man_answers_rating, config, 'man', 'man-rating')
        _rate_answers(question, ai_answers_rating, config, 'man', 'ai-rating')