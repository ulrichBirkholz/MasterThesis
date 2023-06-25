from tsv_utils import get_questions
from tsv_utils import get_answers_per_question
from bert_utils import train_model, AnswersForQuestion
from config import Configuration, BatchSize
import random
import os
import shutil

import argparse
import logging as log

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Train Model with annotated answers')
    parser.add_argument('-mode', default='new', help='Number of training iterations')
    # TODO: we start with 10-20 and evaluate how this affect the rating quality
    parser.add_argument('-epoches', type=int, default=10, help='Number of training iterations')
    return parser.parse_args()

def _train_model_for_question(answers, question, path, args, batch_size):
    finish_marker = f"{path}finished.txt"

    # we started training for this batch
    if os.path.exists(path):
        # the training was finished, nothing left to do
        if os.path.exists(finish_marker):
            return
        else:
            # the answers are randomly selected so we do not continue
            shutil.rmtree(path)
            
    log.debug(f"Training question: {question.question_id} answers: {answers}")
    answer_batch = random.sample(answers, batch_size)
    samples = AnswersForQuestion(question.question_id, question.question, answer_batch)
    train_model(samples, path, args.epoches, args.mode)
    open(finish_marker, "w").close()

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)

    args = setup_args()
    config = Configuration()

    # samples = [{"question":'...', "answers":[answers]}]
    if args.mode == 'new' or args.mode == 'extend':
        questions = get_questions(config.get_questions_path(), False)
        ai_answers = get_answers_per_question(config.get_ai_answers_for_training_path())
        man_answers = get_answers_per_question(config.get_man_answers_for_training_path())

        for question in questions:
            for batch_size in config.get_batch_sizes():
                for id in batch_size.ids:
                    if len(ai_answers[question.question_id]) >= batch_size.size:
                        _train_model_for_question(ai_answers[question.question_id], question,
                                                config.get_trained_bert_model_path(question.question, batch_size.size, id, 'ai'), args, batch_size)
                    else:
                        log.warn(f"Skip batch size {batch_size.size} for automatically created answers, there are not enough: {len(ai_answers[question.question_id])}")
                    if len(man_answers[question.question_id]) >= batch_size.size:
                        _train_model_for_question(man_answers[question.question_id], question,
                                            config.get_trained_bert_model_path(question.question, batch_size.size, id, 'man'), args, batch_size)
                    else:
                        log.warn(f"Skip batch size {batch_size.size} for manually created answers, there are not enough: {len(man_answers[question.question_id])}")
