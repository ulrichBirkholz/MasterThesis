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

previous_answer_batches = []

def _jaccard_similarity(answer_batch_a:List[Answer], answer_batch_b:List[Answer]):
    list1 = [answer.answer_id for answer in answer_batch_a]
    list2 = [answer.answer_id for answer in answer_batch_b]

    intersection = len(set(list1) & set(list2))
    union = len(set(list1) | set(list2))
    return intersection / union

def _is_too_similar(previous_answer_batches, new_answer_batch) -> bool:
    for answer_batch in previous_answer_batches:
        if _jaccard_similarity(answer_batch, new_answer_batch) > 0.2:
            log.error(f"The new batch {[answer.answer_id for answer in new_answer_batch]} is to similar compared to the old one: {[answer.answer_id for answer in answer_batch]}")
            return True
    return False

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Train Model with annotated answers')
    parser.add_argument('-epochs', type=int, default=10, help='Number of training iterations')
    return parser.parse_args()

def _train_model_for_question(answers, question, descriptor_args, args, batch_size, id, descriptor):
    bert_path = config.get_trained_bert_model_path(*descriptor_args)
    xgb_path = config.get_trained_xg_boost_model_path(*descriptor_args)

    # TODO: currently we skip if the file exists in either of folder
    # but we should only skip training the respective model
    for path in [bert_path, xgb_path]:
        # we started training for this batch
        if os.path.exists(path):
            finish_marker = f"{path}description.json"
            # the training was finished, nothing left to do
            if os.path.exists(finish_marker):
                return
            else:
                # the answers are randomly selected so we do not continue
                shutil.rmtree(path)

    answer_batch = random.sample(answers, batch_size.size)
    while _is_too_similar(previous_answer_batches, answer_batch):
        log.error(f"Answer batch is too similar to existing one, number of existing batches: {len(previous_answer_batches)}")
        answer_batch = random.sample(answers, batch_size.size)

    previous_answer_batches.append(answer_batch)

    samples = AnswersForQuestion(question.question_id, question.question, answer_batch)

    log.debug(f"Training question_id: {question.question_id} for batch_size: {len(answer_batch)} with {args.epochs} epochs")
    bert_train_model(samples, bert_path, args.epochs)
    xgb_train_model(samples, xgb_path)

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
                "descriptor": descriptor,
                "epochs": args.epochs,
                "existing_batches": len(previous_answer_batches)
            }, file)

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)
    log.basicConfig(filename='train.log', filemode='w')

    args = setup_args()
    config = Configuration()

    # samples = [{"question":'...', "answers":[answers]}]
    questions = get_questions(config.get_questions_path(), False)
    ai_answers = get_answers_per_question(config.get_ai_answers_for_training_path())
    man_answers = get_answers_per_question(config.get_man_answers_for_training_path())

    total_number_of_modles = 0
    for question in questions:
        for batch_size in config.get_batch_sizes():
            for id in batch_size.ids:
                total_number_of_modles += 2

    train_model = 1
    for question in questions:
        for batch_size in config.get_batch_sizes():
            for id in batch_size.ids:
                log.debug(f"Train model {train_model} of {total_number_of_modles}")
                if len(ai_answers[question.question_id]) >= batch_size.size:
                    descriptor_args = (question.question, batch_size.size, id, 'ai')
                    descriptor = config.get_model_path_descriptor(*descriptor_args)
                    _train_model_for_question(ai_answers[question.question_id], question,
                                            descriptor_args, args, batch_size, id, descriptor)
                else:
                    log.warning(f"Skip batch size {batch_size.size} for automatically created answers, there are not enough: {len(ai_answers[question.question_id])}")
                train_model += 1
                
                log.debug(f"Train model {train_model} of {total_number_of_modles}")
                if len(man_answers[question.question_id]) >= batch_size.size:
                    descriptor_args = (question.question, batch_size.size, id, 'man')
                    descriptor = config.get_model_path_descriptor(*descriptor_args)
                    _train_model_for_question(man_answers[question.question_id], question,
                                        descriptor_args, args, batch_size, id, descriptor)
                else:
                    log.warning(f"Skip batch size {batch_size.size} for manually created answers, there are not enough: {len(man_answers[question.question_id])}")
                train_model += 1
