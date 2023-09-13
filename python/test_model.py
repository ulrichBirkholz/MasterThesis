from tsv_utils import get_questions
from tsv_utils import get_answers_per_question
from tsv_utils import write_rated_answers_tsv
from bert_utils import rate_answers as bert_rate_answers, AnswersForQuestion
from xg_boost_utils import rate_answers as xgb_rate_answers
from config import Configuration
from tsv_utils import Question
from typing import Dict
from config_logger import config_logger

import logging as log
import json
import argparse

def _add_value(cm_matrices, key, matrix, path, iteration=0):
    if iteration > 0:
        key = f"{key}_{iteration}"

    if key in cm_matrices:
        log.error(f"Found duplicated key: {key}")
        # extend the key and retry, we want to see them all
        _add_value(cm_matrices, key, matrix, path, iteration=iteration + 1)
    
    cm_matrices[key] = {
        "cm_matrix": matrix,
        "path": path
    }


def _rate_answers(question: Question, answers_for_question:Dict, config:Configuration, training_data_source:str, test_data_source:str, score_type):
    if question.question_id not in answers_for_question:
            log.info(f"No answers to rate for question: {question}")
            return
    answers = answers_for_question[question.question_id]
 
    cm_matrices = {}
    for batch_size in config.get_batch_sizes():
        for id in batch_size.ids:
            log.debug(f"Rate Answers for question: {question.question_id} with batch size: {batch_size.size}, id: {id}, training_data_source: {training_data_source}")
            
            bert_path = config.get_trained_bert_model_path(question.question, batch_size.size, id, training_data_source)
            bert_rated_answers, bert_cm_matrix = bert_rate_answers(bert_path, AnswersForQuestion(question.question, question.question_id, answers), score_type)
            _add_value(cm_matrices, f"bert_{config.get_model_base_path(question.question, batch_size.size, id, training_data_source)}", bert_cm_matrix, bert_path)

            write_rated_answers_tsv(config.get_test_results_path(f"bert_{training_data_source}", test_data_source, batch_size.size, id), bert_rated_answers, True)

            xgb_path = config.get_trained_xg_boost_model_path(question.question, batch_size.size, id, training_data_source)
            xgb_rated_answers, xgb_cm_matrix = xgb_rate_answers(xgb_path, AnswersForQuestion(question.question, question.question_id, answers), score_type)
            _add_value(cm_matrices, f"xgb_{config.get_model_base_path(question.question, batch_size.size, id, training_data_source)}", xgb_cm_matrix, xgb_path)
            
            write_rated_answers_tsv(config.get_test_results_path(f"xgb_{training_data_source}", test_data_source, batch_size.size, id), xgb_rated_answers, True)

    with open(config.get_path_for_datafile(f"{training_data_source}_{test_data_source}_confusion_matrices.json"), "w") as file:
        json.dump(cm_matrices, file)

def _kv_pairs(string):
    # Convert "key=value" pairs into pairs
    key, value = string.split('=')
    return key, value

# Setup and parse arguments
# example: python -m rate_answers --ai_score_types 5=1 6=1 --man_score_types 5=1 6=1
def setup_args():
    parser = argparse.ArgumentParser(description='Train Model with annotated answers')

    parser.add_argument('--exp', action='store_true', help='Include models trained with samples created by human experts')
    parser.add_argument('--davinci', action='store_true', help='Include models trained with samples created by text-davinci-003')
    parser.add_argument('--turbo', action='store_true', help='Include models trained with samples annotated by gpt-3.5-turbo')
    parser.add_argument('--gpt4', action='store_true', help='Include models trained with samples created by gpt4')

    parser.add_argument('--ai_score_types', metavar='key=value', nargs='+', type=_kv_pairs,
                    help='key-value to defining ai question_id and the respective score_type to be used')
    parser.add_argument('--man_score_types', metavar='key=value', nargs='+', type=_kv_pairs,
                    help='key-value to defining man question_id and the respective score_type to be used')

    return parser.parse_args()

#TODO: refactor score_types
if __name__ == "__main__":
    config_logger(log.DEBUG, "test_model.log")
    args = setup_args()
    config = Configuration()

    questions = get_questions(config.get_questions_path(), False)
    
    all_samples = []

    # the answers, the samples the model was trained with, the samples the model is supposed to rate.
    if args.davinci:
        all_samples.append({
            "answers": get_answers_per_question(config.get_samples_for_testing_path("davinci")),
            "training_data_source": "davinci",
            "test_data_sources": ["davinci-training", "davinci-rating", "turbo-rating", "experts-rating", "gpt4-rating"],
            "score_types": {}
        })
    
    if args.turbo:
        all_samples.append({
            "answers": get_answers_per_question(config.get_samples_for_testing_path("turbo")),
            "training_data_source": "turbo",
            "test_data_sources": ["turbo-training", "davinci-rating", "turbo-rating", "experts-rating", "gpt4-rating"],
            "score_types": {}
        })
    
    if args.gpt4:
        all_samples.append({
            "answers": get_answers_per_question(config.get_samples_for_testing_path("gpt4")),
            "training_data_source": "gpt4",
            "test_data_sources": ["gpt4-training", "davinci-rating", "turbo-rating", "experts-rating", "gpt4-rating"],
            "score_types": {}
        })
    
    if args.exp:
        all_samples.append({
            "answers": get_answers_per_question(config.get_samples_for_testing_path("experts")),
            "training_data_source": "experts",
            "test_data_sources": ["experts-training", "davinci-rating", "turbo-rating", "experts-rating", "gpt4-rating"],
            "score_types": {}
        })

    ai_score_types = {question_id: score_type for question_id, score_type in args.ai_score_types}
    man_score_types = {question_id: score_type for question_id, score_type in args.man_score_types}

    for question in questions:
        # the rating datasets were not used for training but we still relay on the same score_type set to be more comparable
        for samples in all_samples.items():
            for test_source in samples["test_data_sources"]:
                _rate_answers(question, samples["answers"], config, samples["training_data_source"], test_source, samples["score_types"][question.question_id])