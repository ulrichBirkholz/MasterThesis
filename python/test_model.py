from tsv_utils import get_questions
from tsv_utils import get_answers_per_question
from tsv_utils import write_rated_answers_tsv
from bert_utils import test_model as bert_test_model, AnswersForQuestion
from xg_boost_utils import test_model as xgb_test_model
from config import Configuration
from tsv_utils import Question
from typing import Dict, Any, Union, List, Tuple
from config_logger import config_logger

import logging as log
import json
import argparse
from argparse import Namespace


def _add_confusion_matrix(cm_matrices:Dict[str, Dict[str, Union[str, Any]]], key:str, matrix:Any, path:str, iteration=0) -> None:
    """ Safely add a confusion matrix to a dictionary, ensuring there's no key duplication.

    This method adds a confusion matrix to a given dictionary using a unique key. If the specified key already exists
    in the dictionary, it appends an iteration number to the key to avoid data loss and ensures uniqueness.

    Args:
        cm_matrices (Dict[str, Dict[str, Union[str, Any]]]): The dictionary to which the confusion matrix will be added. 
                                                 The value associated with each key is another dictionary containing 
                                                 the confusion matrix and the model path
        key (str): The intended unique identifier for the matrix
        matrix (Any): The confusion matrix to be added
        path (str): The path to the model associated with the confusion matrix
        iteration (int, optional): Used to create a unique key in case of a duplicate. If the key already exists, 
                                   this value is incremented. Defaults to 0.
    """
    if iteration > 0:
        key = f"{key}_{iteration}"

    if key in cm_matrices:
        # this should not happen but if it does we don't want to loose the data
        log.error(f"Found duplicated key: {key}")
        _add_confusion_matrix(cm_matrices, key, matrix, path, iteration=iteration + 1)
    
    cm_matrices[key] = {
        "cm_matrix": matrix,
        "path": path
    }


def _test_model(question: Question, execution:Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]], config:Configuration) -> None:
    """ Test a model with a given set of samples

    Args:
        question (Question): The Question the answers answer
        execution (Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]): _description_
        config (Configuration): _description_
    """
    answers_for_question = execution["answers"]
    training_data_source = execution["training_data_source"]
    test_data_source = execution["test_data_sources"]
    score_type = execution["score_types"][question.question_id]

    if question.question_id not in answers_for_question:
            log.info(f"No answers to rate for question: {question}")
            return
    answers = answers_for_question[question.question_id]
 
    cm_matrices = {}
    for batch_size in config.get_batch_sizes():
        for batch_id in batch_size.ids:
            log.debug(f"Rate Answers for question: {question.question_id} with batch size: {batch_size.size}, batch_id: {batch_id}, training_data_source: {training_data_source}")
            
            bert_path = config.get_trained_bert_model_path(question.question, batch_size.size, batch_id, training_data_source)
            bert_rated_answers, bert_cm_matrix = bert_test_model(bert_path, AnswersForQuestion(question.question, question.question_id, answers), score_type)
            _add_confusion_matrix(cm_matrices, f"bert_{config.get_model_base_path(question.question, batch_size.size, batch_id, training_data_source)}", bert_cm_matrix, bert_path)

            write_rated_answers_tsv(config.get_test_results_path(f"bert_{training_data_source}", test_data_source, batch_size.size, batch_id), bert_rated_answers, True)

            xgb_path = config.get_trained_xg_boost_model_path(question.question, batch_size.size, batch_id, training_data_source)
            xgb_rated_answers, xgb_cm_matrix = xgb_test_model(xgb_path, AnswersForQuestion(question.question, question.question_id, answers), score_type)
            _add_confusion_matrix(cm_matrices, f"xgb_{config.get_model_base_path(question.question, batch_size.size, batch_id, training_data_source)}", xgb_cm_matrix, xgb_path)
            
            write_rated_answers_tsv(config.get_test_results_path(f"xgb_{training_data_source}", test_data_source, batch_size.size, batch_id), xgb_rated_answers, True)

    with open(config.get_path_for_datafile(f"{training_data_source}_{test_data_source}_confusion_matrices.json"), "w") as file:
        json.dump(cm_matrices, file)


# Setup and parse arguments
# example: python -m test_model --score_types_path ./score_types.json --davinci --experts --turbo --gpt4
def setup_args() -> Namespace:
    """ Setup of the execution arguments

    Returns:
        Namespace: arguments to be used
    """
    parser = argparse.ArgumentParser(description='Train Model with annotated answers')

    parser.add_argument("--score_types_path", type=str, required=True, help="Path to the JSON configuration for score types")

    args = parser.parse_args()

    with open(args.score_types_path, 'r') as score_types:
        config = json.load(score_types)

    for key, value in config.items():
        setattr(args, f"score_types_{key}", value)

    return args


def _get_execution(training_data_source:str, test_data_sources:str, file_path:str, score_types:Dict[str, int]) -> Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]:

    return {
        # the answers that are used for testing
        "answers": get_answers_per_question(file_path),
        # source of the samples the model was trained with
        "training_data_source": training_data_source,
        # source of the samples the is tested with
        "test_data_sources": test_data_sources,
        # the score types that will be used according to the calculated distribution
        "score_types": score_types
    }

def _get_executions_for_data_source(model_data_source:str, model_source_types:Dict[str, int], available_data_sources:List[Dict[str, Union[str, Dict[str, int]]]], config:Configuration) -> List[Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]]:
    test_executions = []
    test_executions.append(_get_execution(model_data_source, f"{model_data_source}-training", config.get_samples_for_training_path(model_data_source), model_source_types))
    
    for data_source_info in available_data_sources.items():
        name = data_source_info["name"]
        test_executions.append(_get_execution(model_data_source, f"{name}-testing", config.get_samples_for_testing_path(name), data_source_info["score_types"]))
    
    return test_executions

if __name__ == "__main__":
    config_logger(log.DEBUG, "test_model.log")
    args = setup_args()
    config = Configuration()

    questions = get_questions(config.get_questions_path(), False)

    # TODO: maybe we do not need this, the information is already in the json config file
    available_data_sources = [
        {"name": "davinci", "score_types": args.score_types_davinci},
        {"name": "turbo", "score_types": args.score_types_turbo},
        {"name": "gpt4", "score_types": args.score_types_gpt4},
        {"name": "experts", "score_types": args.score_types_experts}
    ]
    
    test_executions = []
    for data_source in available_data_sources.items():
        test_executions.extend(_get_executions_for_data_source(data_source["name"], data_source["score_types"], available_data_sources, config))

    for question in questions:
        # the rating datasets were not used for training but we still relay on the same score_type set to be more comparable
        for execution in test_executions.items():
            _test_model(question, execution, config)