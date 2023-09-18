from tsv_utils import get_questions, Answer, Question
from tsv_utils import get_answers_per_question
from bert_utils import train_model as bert_train_model, AnswersForQuestion
from xg_boost_utils import train_model as xgb_train_model
from config import Configuration
import random
import os
import shutil
import json

import argparse
from argparse import Namespace
import logging as log

from typing import List, Dict, Tuple, Union

from config_logger import config_logger

previous_answer_batches = []

def _jaccard_similarity(answer_batch_a:List[Answer], answer_batch_b:List[Answer]) -> float:
    """ Calculates the intersection of two given lists of Answers

    Args:
        answer_batch_a (List[Answer]): First list
        answer_batch_b (List[Answer]): Second list

    Returns:
        float: jaccard_similarity
    """
    list1 = [answer.answer_id for answer in answer_batch_a]
    list2 = [answer.answer_id for answer in answer_batch_b]

    intersection = len(set(list1) & set(list2))
    union = len(set(list1) | set(list2))
    return intersection / union

# We want to ensure a certain variance within the datasets to be able to observe how different answers affect the efficiency of the model,
#   this is at this point considered more valuable than pure randomization
def _is_selection_valid(previous_answer_batches:List[List[Answer]], new_answer_batch:List[Answer], score_type:int) -> bool:
    """ Ensures that the intersection between the newly selected List is above a certain threshold and that each
    category or possible score is present in the selection.

    Args:
        previous_answer_batches (List[List[Answer]]): All previouly selected Lists of Answers
        new_answer_batch (List[Answer]): Newly selected List of Answers
        score_type (int): The score type to be used for the evaluation (either 1 or 2)

    Returns:
        bool: True if the selection is valid, False otherwise
    """
    for answer_batch in previous_answer_batches:
        similarity = _jaccard_similarity(answer_batch, new_answer_batch)

        if len(new_answer_batch) <= 800:
            max_similarity = 0.2
        else:
            # larger batches naturally have a higher similarity
            max_similarity = 0.52

        if similarity > max_similarity:
            # log more information for smaller batches
            if len(new_answer_batch) <= 400:
                log.error(f"The new batch {[answer.answer_id for answer in new_answer_batch]} with len: {len(new_answer_batch)} is to similar: {similarity} compared to the old one: {[answer.answer_id for answer in answer_batch]} with len: {len(answer_batch)}")
            else:
                # it is not useful to print too large datasets, the entire line won't be displayed anyway
                log.error(f"The new batch with len: {len(new_answer_batch)} is to similar: {similarity} compared to the old one with len: {len(answer_batch)}")
            return True

    # ensure the presence of at leased one answer per score_type
    score_types = set([getattr(answer, f'score_{score_type}') for answer in new_answer_batch])
    if len(score_types) < 4:
        log.error(f"One answer of each category (0 - 3) must be present: {score_types}")
        return True
    
    return False

# Setup and parse arguments
# example: python -m train_model --score_types_path ./score_types.json --davinci --exp --turbo --gpt4
def setup_args() -> Namespace:
    """ Setup of the execution arguments

    Returns:
        Namespace: arguments to be used
    """
    parser = argparse.ArgumentParser(description='Train Model with annotated answers')

    parser.add_argument('--exp', action='store_true', help='Include samples created by human experts')
    parser.add_argument('--davinci', action='store_true', help='Include samples created by text-davinci-003')
    parser.add_argument('--turbo', action='store_true', help='Include samples annotated by gpt-3.5-turbo')
    parser.add_argument('--gpt4', action='store_true', help='Include samples created by gpt4')

    parser.add_argument('-epochs', type=int, default=10, help='Number of training iterations')
    parser.add_argument("--score_types_path", type=str, required=True, help="Path to the JSON configuration for score types")

    args = parser.parse_args()

    with open(args.score_types_path, 'r') as json_score_types:
        score_types = json.load(json_score_types)

    for key, value in score_types.items():
        setattr(args, f"score_types_{key}", value)

    return args


def _does_model_exist(paths:List[str]) -> bool:
    """ Checks if a given List of model, identified by their paths, are fully trained
    The decision about this is based on the existence of the 'description.json' file,
    which is created after the model was trained.

    In case a model is not considered to be fully trained, the folder is deleted.

    Args:
        paths (List[str]): List of paths leading to the models folders

    Returns:
        bool: True if all models are fully trained, False otherwise
    """
    markers = [f"{path}description.json" for path in paths]
    markers_exist = [os.path.exists(marker) for marker in markers]

    # If any of the marker_does not exist, delete all models for this question
    if not all(markers_exist):
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
        return False

    return True


def _get_random_answers(answers:List[Answer], batch_size:int) -> List[Answer]:
    """ Select a random Set of Answers from a given List

    Args:
        answers (List[Answer]): The List of Answers to select form
        batch_size (int): The number of Answers to be selected

    Returns:
        List[Answer]: Randomly selected Answers
    """
    random.shuffle(answers)
    return answers[:batch_size]


def _train_model_for_question(answers:List[Answer], question:Question, path_args:Tuple[str, int, str, str], args:Namespace, batch_size:int, batch_id:str, base_path:str, score_type:int) -> None:
    """ Trains an BERT and XG-Boost model based on the given Samples

    Args:
        answers (List[Answer]): Sample Answers
        question (Question): Sample Question
        path_args (Tuple[str, int, str, str]): Arguments to determine the folder path to the trained models
        args (_type_): Execution Arguments
        batch_size (int): Number of Sample Answers the model will be trained with
        batch_id (str): Id of the Variance we Train
        base_path (str): The internal path to the model
        score_type (int): The score type to be used for the evaluation (either 1 or 2)
    """
    assert score_type == 1 or score_type == 2, f"The used score type: {score_type} is invalid"
    bert_path = config.get_trained_bert_model_path(*path_args)
    xgb_path = config.get_trained_xg_boost_model_path(*path_args)

    if _does_model_exist([bert_path, xgb_path]):
        return

    answer_batch = _get_random_answers(answers, batch_size)

    while _is_selection_valid(previous_answer_batches, answer_batch, score_type):
        log.error(f"Answer batch is too similar to existing one, number of existing batches: {len(previous_answer_batches)}")
        answer_batch = _get_random_answers(answers, batch_size)

    previous_answer_batches.append(answer_batch)

    samples = AnswersForQuestion(question.question_id, question.question, answer_batch)

    log.debug(f"Training question_id: {question.question_id} for batch_size: {len(answer_batch)} with {args.epochs} epochs")
    bert_train_model(samples, bert_path, args.epochs, score_type)
    xgb_train_model(samples, xgb_path, score_type)

    for path in [bert_path, xgb_path]:
        finish_marker = f"{path}description.json"
        with open(finish_marker, "w", encoding='utf-8') as file:
            json.dump({
                "answer_batch": [{
                    "answer": answer.answer,
                    "answer_id": answer.answer_id,
                    "score_1": answer.score_1,
                    "score_2": answer.score_2
                } for answer in answer_batch],
                "question_id": question.question_id,
                "question": question.question,
                "batch_size": batch_size,
                "batch_variant_id": batch_id,
                "base_path": base_path,
                "epochs": args.epochs,
                "existing_batches": len(previous_answer_batches)
            }, file)

def _setup_training_for(training_data_source:str, score_types:Dict[str, int], config:Configuration) -> Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]:
    """ Combines all relevant information about the Samples for one Execution

    Args:
        training_data_source (str): The source of the Samples, the model will be trained with
        score_types (Dict[str, int]): The score types (either 1 or 2) to be used, this is individual per Question
        config (Configuration): Allows access to the projects central configuration

    Returns:
        Dict[str, Union[AnswersForQuestion, str, Dict[str, int]]]: The combined Information
    """
    return {
        "answers": get_answers_per_question(config.get_samples_for_training_path(training_data_source)),
        "source": training_data_source,
        "score_types": score_types
    }
    

if __name__ == "__main__":
    config_logger(log.DEBUG, "train.log")
    args = setup_args()
    config = Configuration()

    # samples = [{"question":'...', "answers":[answers]}]
    questions = get_questions(config.get_questions_path(), False)

    trainings = []
    if args.davinci:
        trainings.append(_setup_training_for("davinci", args.score_types_davinci, config))
    
    if args.turbo:
        trainings.append(_setup_training_for("turbo", args.score_types_turbo, config))
    
    if args.gpt4:
        trainings.append(_setup_training_for("gpt4", args.score_types_gpt4, config))
    
    if args.exp:
        trainings.append(_setup_training_for("experts", args.score_types_experts, config))

    total_number_of_models = 0
    for question in questions:
        for batch in config.get_batches():
            for batch_id in batch.ids:
                total_number_of_models += len(trainings) * 2

    train_model = 1
    for question in questions:
        for batch in config.get_batches():
            for batch_id in batch.ids:
                for training in trainings:
                    answers = training["answers"]

                    log.debug(f"Train model {train_model} of {total_number_of_models}")
                    if len(answers[question.question_id]) >= batch.size:
                        path_args = (question.question, batch.size, batch_id, training["source"])
                        base_path = config.get_relative_model_path(*path_args)
                        _train_model_for_question(answers[question.question_id], question,
                                                path_args, args, batch.size, batch_id, base_path, training["score_types"][question.question_id])
                    else:
                        log.warning(f"Skip batch size {batch.size} for automatically created answers, there are not enough: {len(answers[question.question_id])}")
                    # BERT + XG_Boost
                    train_model += 2
