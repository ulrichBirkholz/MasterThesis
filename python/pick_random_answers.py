import random
from tsv_utils import get_answers_per_question, write_answers_tsv, Answer
from config import Configuration
from typing import List

def _split_array(answers:List[Answer], training_size):
    answers_for_training = random.sample(answers, training_size)  # Select training_size random elements used to train the model
    answers_for_rating = [answer for answer in answers if answer not in answers_for_training] # The rest is for rating

    return answers_for_training, answers_for_rating

def _split_answers_for_question(answers_per_question, mode:str, config:Configuration):
    for question_id, answers in answers_per_question.items():
        if mode == 'ai':
            answers_for_training, answers_for_rating = _split_array(answers, 3200)
            write_answers_tsv(config.get_ai_answers_for_training_path(), [answers_for_training], True)
            write_answers_tsv(config.get_ai_answers_to_rate_path(), [answers_for_rating], True)
        else:
            answers_for_training, answers_for_rating = _split_array(answers, 1600)
            write_answers_tsv(config.get_man_answers_for_training_path(), [answers_for_training], True)
            write_answers_tsv(config.get_man_answers_to_rate_path(), [answers_for_rating], True)


if __name__ == "__main__":
    config = Configuration()
    ai_answers = get_answers_per_question(config.get_ai_answers_path())
    man_answers = get_answers_per_question(config.get_man_answers_path())

    _split_answers_for_question(ai_answers, 'ai', config)
    #_split_answers_for_question(man_answers, 'man', config)

    
    

    