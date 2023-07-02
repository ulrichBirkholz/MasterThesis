from tsv_utils import get_questions
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

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Train Model with annotated answers')
    parser.add_argument('-epoches', type=int, default=15, help='Number of training iterations')
    return parser.parse_args()

def _train_model_for_question(answers, question, descriptor_args, args, batch_size, id, descriptor):
    bert_path = config.get_trained_bert_model_path(*descriptor_args)
    xgb_path = config.get_trained_xg_boost_model_path(*descriptor_args)

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
            
    log.debug(f"Training question: {question.question_id} answers: {answers}")
    answer_batch = random.sample(answers, batch_size.size)
    samples = AnswersForQuestion(question.question_id, question.question, answer_batch)
    bert_train_model(samples, bert_path, args.epoches)
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
                "descriptor": descriptor
            }, file)

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)

    args = setup_args()
    config = Configuration()

    # samples = [{"question":'...', "answers":[answers]}]
    questions = get_questions(config.get_questions_path(), False)
    ai_answers = get_answers_per_question(config.get_ai_answers_for_training_path())
    man_answers = get_answers_per_question(config.get_man_answers_for_training_path())

    for question in questions:
        for batch_size in config.get_batch_sizes():
            for id in batch_size.ids:
                if len(ai_answers[question.question_id]) >= batch_size.size:
                    descriptor_args = (question.question, batch_size.size, id, 'ai')
                    descriptor = config.get_model_path_descriptor(*descriptor_args)
                    _train_model_for_question(ai_answers[question.question_id], question,
                                            descriptor_args, args, batch_size, id, descriptor)
                else:
                    log.warning(f"Skip batch size {batch_size.size} for automatically created answers, there are not enough: {len(ai_answers[question.question_id])}")
                
                if len(man_answers[question.question_id]) >= batch_size.size:
                    descriptor_args = (question.question, batch_size.size, id, 'man')
                    descriptor = config.get_model_path_descriptor(*descriptor_args)
                    _train_model_for_question(man_answers[question.question_id], question,
                                        descriptor_args, args, batch_size, id, descriptor)
                else:
                    log.warning(f"Skip batch size {batch_size.size} for manually created answers, there are not enough: {len(man_answers[question.question_id])}")
