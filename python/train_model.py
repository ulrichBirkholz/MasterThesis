from tsv_utils import get_questions, Answer
from tsv_utils import get_answers_per_question
from bert_utils import train_model as bert_train_model, AnswersForQuestion
from xg_boost_utils import train_model as xgb_train_model
from config import Configuration
import random
import os
import shutil
import json

import argparse
import logging as log

from typing import List

from config_logger import config_logger

previous_answer_batches = []

def _jaccard_similarity(answer_batch_a:List[Answer], answer_batch_b:List[Answer]):
    list1 = [answer.answer_id for answer in answer_batch_a]
    list2 = [answer.answer_id for answer in answer_batch_b]

    intersection = len(set(list1) & set(list2))
    union = len(set(list1) | set(list2))
    return intersection / union

def _is_too_similar(previous_answer_batches, new_answer_batch, score_type:int) -> bool:
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


def _kv_pairs(string):
    # Convert "key=value" pairs into pairs
    key, value = string.split('=')
    return key, value


# Setup and parse arguments
# example: python -m train_model --ai_score_types 5=1 6=1 --man_score_types 5=1 6=1 --exp --davinci --turbo --gpt4
def setup_args():
    parser = argparse.ArgumentParser(description='Train Model with annotated answers')

    parser.add_argument('--exp', action='store_true', help='Include samples created by human experts')
    parser.add_argument('--davinci', action='store_true', help='Include samples created by text-davinci-003')
    parser.add_argument('--turbo', action='store_true', help='Include samples annotated by gpt-3.5-turbo')
    parser.add_argument('--gpt4', action='store_true', help='Include samples created by gpt4')
    parser.add_argument('--ai_score_types', metavar='key=value', nargs='+', type=_kv_pairs,
                    help='key-value to defining ai question_id and the respective score_type to be used')
    parser.add_argument('--man_score_types', metavar='key=value', nargs='+', type=_kv_pairs,
                    help='key-value to defining man question_id and the respective score_type to be used')

    parser.add_argument('-epochs', type=int, default=10, help='Number of training iterations')
    return parser.parse_args()


def _does_model_exist(paths):
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
    random.shuffle(answers)
    return answers[:batch_size]


def _train_model_for_question(answers, question, path_args, args, batch_size, id, base_path, score_type):
    bert_path = config.get_trained_bert_model_path(*path_args)
    xgb_path = config.get_trained_xg_boost_model_path(*path_args)

    if _does_model_exist([bert_path, xgb_path]):
        return

    answer_batch = _get_random_answers(answers, batch_size)

    # NOTE: we observed several similarities of 1.0, which translates to the selection of identical answer batches
    # We want to ensure a certain variance within the datasets to be able to observe how different answers affect the efficiency of the model,
    #   this is at this point considered more valuable than pure randomization
    while _is_too_similar(previous_answer_batches, answer_batch, score_type):
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
                "batch_size": batch_size.size,
                "batch_variant_id": id,
                "base_path": base_path,
                "epochs": args.epochs,
                "existing_batches": len(previous_answer_batches)
            }, file)

#TODO: refactor score_types
if __name__ == "__main__":
    config_logger(log.DEBUG, "train.log")
    args = setup_args()
    config = Configuration()

    # samples = [{"question":'...', "answers":[answers]}]
    questions = get_questions(config.get_questions_path(), False)

    all_samples = []
    if args.davinci:
        all_samples.append({
            "answers": get_answers_per_question(config.get_samples_for_training_path("davinci")),
            "source": "davinci",
            "score_types": {}
        })
    
    if args.turbo:
        all_samples.append({
            "answers": get_answers_per_question(config.get_samples_for_training_path("turbo")),
            "source": "turbo",
            "score_types": {}
        })
    
    if args.gpt4:
        all_samples.append({
            "answers": get_answers_per_question(config.get_samples_for_training_path("gpt4")),
            "source": "gpt4",
            "score_types": {}
        })
    
    if args.exp:
        all_samples.append({
            "answers": get_answers_per_question(config.get_samples_for_training_path("experts")),
            "source": "experts",
            "score_types": {}
        })

    total_number_of_models = 0
    for question in questions:
        for batch_size in config.get_batch_sizes():
            for id in batch_size.ids:
                total_number_of_models += len(all_samples)

    ai_score_types = {question_id: score_type for question_id, score_type in args.ai_score_types}
    man_score_types = {question_id: score_type for question_id, score_type in args.man_score_types}

    train_model = 1
    for question in questions:
        for batch_size in config.get_batch_sizes():
            for id in batch_size.ids:
                for samples in all_samples.items():
                    answers = samples["answers"]

                    log.debug(f"Train model {train_model} of {total_number_of_models}")
                    if len(answers[question.question_id]) >= batch_size.size:
                        path_args = (question.question, batch_size.size, id, samples["source"])
                        base_path = config.get_model_base_path(*path_args)
                        _train_model_for_question(answers[question.question_id], question,
                                                path_args, args, batch_size, id, base_path, samples["score_types"][question.question_id])
                    else:
                        log.warning(f"Skip batch size {batch_size.size} for automatically created answers, there are not enough: {len(answers[question.question_id])}")
                    train_model += 1
