from tsv_utils import get_questions
from tsv_utils import get_answers_per_question
from tsv_utils import write_rated_answers_tsv
from bert_utils import test_model as bert_test_model, AnswersForQuestion
from xg_boost_utils import test_model as xgb_test_model
from config import Configuration
from tsv_utils import Question
from typing import Dict, Any, List, Union
from config_logger import config_logger

import logging as log
import json
import argparse
from argparse import Namespace
import os


def _add_confusion_matrix(cm_matrices:Dict[str, Dict[str, Union[str, Any]]], key:str, matrix:Any, path:str, iteration=0) -> None:
    """ Safely add a confusion matrix to a dictionary, ensuring there's no key duplication.

    This method adds a confusion matrix to a given dictionary using a unique key. If the specified key already exists
    in the dictionary, it appends an iteration number to the key to avoid data loss and ensures uniqueness.

    Args:
        cm_matrices (Dict[str, Dict[str, Union[str, Any]]]]): The dictionary to which the confusion matrix will be added. 
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
    """ Test a model using the provided execution configuration and save the results.

    This function takes in a specific question, the execution details (which consolidates 
    various pieces of information relevant for a model's execution), and a configuration object.
    It tests both the BERT and XGBoost models, logs debug and info messages, and writes 
    rated answers to TSV files. It also constructs confusion matrices for each test and saves them as JSON.

    Args:
        question (Question): The question for which answers need to be rated
        execution (Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]): Consolidation of various pieces of information relevant for a model's execution in a single dictionary
        config (Configuration): Allows access to the projects central configuration
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
    for batch in config.get_batches():
        for batch_id in batch.ids:
            log.debug(f"Rate Answers for question: {question.question_id} with batch size: {batch.size}, batch_id: {batch_id}, training_data_source: {training_data_source}")
            
            bert_path = config.get_trained_bert_model_path(question.question_id, batch.size, batch_id, training_data_source)
            bert_rated_answers, bert_cm_matrix = bert_test_model(bert_path, AnswersForQuestion(question.question, question.question_id, answers), score_type)
            _add_confusion_matrix(cm_matrices, f"bert_{config.get_relative_model_path(question.question_id, batch.size, batch_id, training_data_source)}", bert_cm_matrix, bert_path)

            write_rated_answers_tsv(config.get_test_results_path("bert", training_data_source, test_data_source, batch.size, batch_id), bert_rated_answers, True)

            xgb_path = config.get_trained_xg_boost_model_path(question.question_id, batch.size, batch_id, training_data_source)
            xgb_rated_answers, xgb_cm_matrix = xgb_test_model(xgb_path, AnswersForQuestion(question.question, question.question_id, answers), score_type)
            _add_confusion_matrix(cm_matrices, f"xgb_{config.get_relative_model_path(question.question_id, batch.size, batch_id, training_data_source)}", xgb_cm_matrix, xgb_path)
            
            write_rated_answers_tsv(config.get_test_results_path("xgb", training_data_source, test_data_source, batch.size, batch_id), xgb_rated_answers, True)

    with open(config.get_path_for_datafile(f"{training_data_source}_{test_data_source}_{question.question_id}_confusion_matrices.json"), "w") as file:
        json.dump(cm_matrices, file)


# Setup and parse arguments
# example: python -m test_model --score_types_path ./score_types.json
def setup_args() -> Namespace:
    """ Setup of the execution arguments

    Returns:
        Namespace: arguments to be used
    """
    parser = argparse.ArgumentParser(description='Test Model with annotated answers')

    parser.add_argument("--score_types_path", type=str, required=True, help="Path to the JSON configuration for score types")

    args = parser.parse_args()

    with open(args.score_types_path, 'r') as json_score_types:
        score_types = json.load(json_score_types)

    setattr(args, "score_types", score_types)

    return args


def _get_execution(training_data_source:str, test_data_sources:str, file_path:str, score_types:Dict[str, int]) -> Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]:
    """ Consolidates various pieces of information relevant for a model's execution into a single dictionary

    Args:
        training_data_source (str): The source of the Samples, the model has been trained with
        test_data_sources (str): The source of the Samples, the model will be tested with
        file_path (str): Path to the file containing the samples
        score_types (Dict[str, int]): The score types (either 1 or 2) to be used, this is individual per Question

    Returns:
        Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]: A dictionary consolidating all the provided
        information. It includes:
            - answers: Grouped answers by their question derived from the file_path
            - training_data_source: Source of the training samples
            - test_data_sources: Source of the testing samples
            - score_types: Score types per question
    """
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


def _get_executions_for_data_source(model_data_source:str, model_score_types:Dict[str, int], available_data_sources:List[Dict[str, Union[str, Dict[str, int]]]], config:Configuration) -> List[Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]]:
    """ Generate a list of consolidated execution configurations for a given model and available data sources.

    This function creates a consolidated set of execution details for each available dataset.
    The details include information about the source of the training samples, the score type
    the model has been trained with, and configurations specific to available data sources

    Args:
        model_data_source (str): The source of the Samples, the model has been trained with
        model_score_types (Dict[str, int]): The score type the model has been trained with
        available_data_sources (List[Dict[str, Union[str, Dict[str, int]]]]): 
            A list containing information about all available data sources and their respective 
            scoring types
        config (Configuration):Allows access to the projects central configuration

    Returns:
        List[Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]]:
            A list of consolidated execution configurations for each available data source
    """    
    test_executions = []
    test_executions.append(_get_execution(model_data_source, f"{model_data_source}-training", config.get_samples_for_training_path(model_data_source), model_score_types))
    
    for data_source_info in available_data_sources:
        name = data_source_info["name"]
        test_executions.append(_get_execution(model_data_source, f"{name}-testing", config.get_samples_for_testing_path(name), data_source_info["score_types"]))
    
    return test_executions


def _delete_file(file:str) -> None:
    """ Deletes a given file if it exists

    Args:
        file (str): Path to the file
    """
    # cleanup
    if os.path.exists(file):
        os.remove(file)


def _cleanup(config:Configuration, executions:List[Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]]) -> None:
    """ Ensures that none of the files, created by this module, already exist

    Args:
        config (Configuration): Allows access to the projects central configuration
        execution (List[Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]]): Consolidations of various pieces of information relevant for the model's executions
    """

    data_root_path = config.get_datafile_root_path()
    for data_file in os.listdir(data_root_path):
        if data_file.endswith("_confusion_matrices.json"):
            full_path = os.path.join(data_root_path, data_file)
            os.remove(full_path)

    for batch in config.get_batches():
        for batch_id in batch.ids:
            for execution in executions:
                _delete_file(config.get_test_results_path("xgb", execution["training_data_source"], execution["test_data_sources"], batch.size, batch_id))
                _delete_file(config.get_test_results_path("bert", execution["training_data_source"], execution["test_data_sources"], batch.size, batch_id))


if __name__ == "__main__":
    config_logger(log.DEBUG, "test_model.log")
    args = setup_args()
    config = Configuration()

    questions = get_questions(config.get_questions_path(), False)
    available_data_sources = [{"name": key, "score_types": value} for key, value in args.score_types.items()]
    
    test_executions = []
    for data_source in available_data_sources:
        test_executions.extend(_get_executions_for_data_source(data_source["name"], data_source["score_types"], available_data_sources, config))

    _cleanup(config, test_executions)
    for question in questions:
        # the rating datasets were not used for training but we still relay on the same score_type set to be more comparable
        for execution in test_executions:
            _test_model(question, execution, config)