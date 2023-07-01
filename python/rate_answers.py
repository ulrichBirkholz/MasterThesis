from tsv_utils import get_questions
from tsv_utils import get_answers_per_question
from tsv_utils import write_rated_answers_tsv
from bert_utils import rate_answers as bert_rate_answers, AnswersForQuestion
from xg_boost_utils import rate_answers as xgb_rate_answers
from config import Configuration
from tsv_utils import Question
from typing import Dict
import random

import logging as log
import json

def _rate_answers(question: Question, answers_for_question:Dict, config:Configuration, model_descriptor:str, answer_descriptor:str):
    if question.question_id not in answers_for_question:
            log.info(f"No answers to rate for question: {question}")
            return
    answers = answers_for_question[question.question_id]

    # we rate max. 1000 answers
    if len(answers) > 1000:
          answers = random.sample(answers, 1000)
 
    cm_matrices = {}
    for batch_size in config.get_batch_sizes():
        for id in batch_size.ids:
            log.debug(f"Rate Answers for question: {question.question_id} with batch size: {batch_size.size}, id: {id}, model_descriptor: {model_descriptor}")
            
            bert_rated_answers, bert_cm_matrix = bert_rate_answers(config.get_trained_bert_model_path(question.question, batch_size.size, id, model_descriptor), AnswersForQuestion(question.question, question.question_id, answers))
            cm_matrices[f"bert_{config.get_model_path_descriptor(question.question, batch_size.size, id, model_descriptor)}"] = bert_cm_matrix

            xgb_rated_answers, xgb_cm_matrix = xgb_rate_answers(config.get_trained_bert_model_path(question.question, batch_size.size, id, model_descriptor), AnswersForQuestion(question.question, question.question_id, answers))
            cm_matrices[f"xgb_{config.get_model_path_descriptor(question.question, batch_size.size, id, model_descriptor)}"] = xgb_cm_matrix
            
            write_rated_answers_tsv(config.get_rated_answers_path(f"bert_{model_descriptor}", answer_descriptor, batch_size.size, id), bert_rated_answers, True)
            write_rated_answers_tsv(config.get_rated_answers_path(model_descriptor, answer_descriptor, batch_size.size, id), xgb_rated_answers, True)
    

    with open(config.get_path_for_datafile(f"{model_descriptor}_{answer_descriptor}_confusion_matrices.json"), "w") as file:
        json.dump(cm_matrices, file)

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