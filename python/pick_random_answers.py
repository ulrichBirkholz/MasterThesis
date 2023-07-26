import random
from tsv_utils import get_answers_per_question, write_answers_tsv, Answer
from config import Configuration
from collections import Counter
from typing import List
import logging as log
from config_logger import config_logger

# {'category', percent} -> {'0': 10, '1': 50, '2': 15, '3': 25}
def _calculate_distribution(answers:List[Answer]):
    category_counts = Counter([answer.score_2 for answer in answers])
    total_count = len(answers)
    return {category: round((count / total_count) * 100) for category, count in category_counts.items()}


def _get_adjust_distribution(ai_answers, man_answers_distribution):
    total_count = len(ai_answers)

    category_items = {category: [answer for answer in ai_answers if answer.score_2 == category] for category in man_answers_distribution}

    lowest_category_ratio = min(len(category_items[category]) / round((target_percent / 100) * total_count) 
                                for category, target_percent in man_answers_distribution.items())

    log.debug(f"Lowest category ratio: {lowest_category_ratio}")
    adjusted_distribution = []

    for category, target_percent in man_answers_distribution.items():
        num_in_category = round((target_percent / 100) * total_count * lowest_category_ratio)
        adjusted_distribution += random.sample(category_items[category], num_in_category)

    return adjusted_distribution

def _split_array(answers:List[Answer], training_size):
    distribution = _calculate_distribution(answers)

    training_factor = training_size / len(answers)
    rating_factor = 1 - training_factor

    category_items = {category: [answer for answer in ai_answers if answer.score_2 == category] for category in man_answers_distribution}

    answers_for_training = []
    answers_for_rating = []
    for category, target_percent in distribution.items():
        sample_size_training = len(category_items[category]) * training_factor
        answers_for_training.extend(random.sample(category_items[category], sample_size_training))

        sample_size_rating = len(category_items[category]) * rating_factor
        answers_for_rating.extend(random.sample(category_items[category], sample_size_rating))

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
    config_logger(log.DEBUG, "pick_answers.log")
    ai_answers_per_question = get_answers_per_question(config.get_ai_answers_path())
    man_answers_per_question = get_answers_per_question(config.get_man_answers_path())

    ai_answers_redistributed_per_question = {}
    for question_id, man_answers in man_answers_per_question.items():
        ai_answers = ai_answers_per_question[question_id]

        man_answers_distribution = _calculate_distribution(man_answers)
        log.debug(f"Calculated the following distribution: {man_answers_distribution} for question: {question_id}")

        ai_answers_redistributed = _get_adjust_distribution(ai_answers, man_answers_distribution)

        redistributed_count = len(ai_answers_redistributed)
        original_count = len(ai_answers)

        if redistributed_count != original_count:
            log.debug(f"""Updated distribution of AI answers from: {original_count} to: {redistributed_count};
this reduced the total answers from: {_calculate_distribution(ai_answers)}, to: {_calculate_distribution(ai_answers_redistributed)}""")

        ai_answers_redistributed_per_question[question_id] = ai_answers_redistributed

    _split_answers_for_question(ai_answers_redistributed_per_question, 'ai', config)
    _split_answers_for_question(man_answers, 'man', config)

    
    

    