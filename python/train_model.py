from tsv_utils import get_questions
from tsv_utils import get_answers_per_question
from bert_utils import train_model
from config import get_config

import argparse
import logging as log

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Train Model with annotated answers')
    parser.add_argument('-mode', default='new', help='Number of training iterations')
    # TODO: we start with 10-20 and evaluate how this affect the rating quality
    parser.add_argument('-epoches', type=int, default=10, help='Number of training iterations')
    return parser.parse_args()

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)

    args = setup_args()
    config = get_config()

    # samples = [{"question":'...', "answers":[answers]}]
    samples = []
    if args.mode == 'new' or args.mode == 'extend':
        questions = get_questions(config["data_path"] + config["questions"], False)
        answers = get_answers_per_question(config["data_path"] + config["answers"])

        for question in questions:
            answers_for_question = answers[question[-1]]
            log.debug(f"Training question: {question[0]} answers: {answers_for_question}")

            samples.append({"question": question[0], "answers": answers_for_question})

    train_model(samples, config["model_path"], args.epoches, args.mode)