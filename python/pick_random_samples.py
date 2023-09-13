import random
from tsv_utils import get_answers_per_question, write_answers_tsv, Answer
from config import Configuration
from collections import Counter
from typing import List
import logging as log
from config_logger import config_logger
import os
import argparse

class DataWriter:
    def __init__(self):
        self.lines = []

    def add_line(self, description, information):
        self.lines.append(f"{description}: {information}")
    
    def write_to_file(self, path):
        self.lines.sort()
        with open(path, 'a') as file:
            file.write('\n'.join(self.lines))

_data_writer = DataWriter()

# {'category', percent} -> {'0': 10, '1': 50, '2': 15, '3': 25}
def _calculate_distribution(answers:List[Answer], score_type:int, total:bool = False):
    category_counts = Counter([getattr(answer, f'score_{score_type}') for answer in answers])
    if total:
        return {category: category_counts[category] for category in sorted(category_counts)}
    else:
        total_count = len(answers)
        return {category: round((category_counts[category] / total_count) * 100) for category in sorted(category_counts)}

def _adjust_distribution(ai_answers, experts_answers_distribution, score_type:int, min_size:int=0):
    total_count = len(ai_answers)
    assert min_size < total_count, f"The minimal required result size: {min_size} is larger than the total number of answers: {total_count}"

    category_items = {category: [answer for answer in ai_answers if getattr(answer, f'score_{score_type}') == category] for category in experts_answers_distribution}

    lowest_category_ratio = min(len(category_items[category]) / round((target_percent / 100) * total_count) 
                                for category, target_percent in experts_answers_distribution.items())

    log.debug(f"Lowest category ratio: {lowest_category_ratio}")

    result = []
    tolerance = 0
    # increase tolerance until we reach the desired number of answers
    while (len(result) < min_size):
        result.clear()
        for category, target_percent in experts_answers_distribution.items():
            number_of_samples = len(category_items[category])
            number_to_pick = round((target_percent / 100) * total_count * lowest_category_ratio) + tolerance

            if number_to_pick > number_of_samples:
                number_to_pick = number_of_samples

            result.extend(random.sample(category_items[category], number_to_pick))
        tolerance += 1

    return result


def _select_answers_for_category(category_item, factor):
    category_len = len(category_item)
    sample_size = round(category_len * factor)
    log.debug(f"About to pick {sample_size} random answers form {category_len} samples")
    result = random.sample(category_item, sample_size)
    for answer in result:
        category_item.remove(answer)
    return result


def _split_answers(all_answers:List[Answer], distributed_answers:List[Answer], score_type, training_size):
    assert len(distributed_answers) >= training_size + 100, f"""The distribution is too small!
We need at leased {training_size} answers for training + 100 answers for rating, but we have only {len(distributed_answers)}"""

    # we split the list of answers based on its current distribution and the defined training_size
    distribution = _calculate_distribution(distributed_answers, score_type)

    training_factor =  training_size / len(distributed_answers)

    category_items = {category: [answer for answer in distributed_answers if getattr(answer, f'score_{score_type}') == category] for category in distribution}

    answers_for_training = []

    # the category fits already to the desired distribution
    for category, target_percent in distribution.items():
        category_item = category_items[category]
        answers_for_training.extend(_select_answers_for_category(category_item, training_factor))

    # rounding requires sometimes a fill up
    if len(answers_for_training) < training_size:
        unused_answers = [answer for answer in distributed_answers if answer not in answers_for_training]
        answers_for_training.extend(random.sample(unused_answers, training_size - len(answers_for_training)))
    
    answers_for_rating = [answer for answer in all_answers if answer not in answers_for_training]
    return answers_for_training, answers_for_rating


def _split_samples_for_question(answers_redistributed_per_question, source:str, training_size:int, config:Configuration):
    training_path = config.get_samples_for_training_path(source)
    rating_path = config.get_samples_to_rate_path(source)

    for question_id, distribution in answers_redistributed_per_question.items():
        log.debug(f"Split {source} answers for question: {question_id}")

        score_type = distribution["score_type"]
        answers = distribution["answers"]

        all_answers = distribution["all_answers"]
        _data_writer.add_line(f"{source} answers for question: {question_id} use score_type", score_type)
        _data_writer.add_line(f"Distribution in % of all {source} answers for question: {question_id}", _calculate_distribution(all_answers, score_type))
        _data_writer.add_line(f"Distribution of all {source} answers for question: {question_id}", _calculate_distribution(all_answers, score_type, True))

        _data_writer.add_line(f"Distribution in % of used {source} answers for question: {question_id}", _calculate_distribution(answers, score_type))
        _data_writer.add_line(f"Distribution of used {source} answers for question: {question_id}", _calculate_distribution(answers, score_type, True))

        _data_writer.add_line(f"Number of all {source} answers for question: {question_id}", len(all_answers))
        log.debug(f"distribution contains: {len(answers)} answers")

        answers_for_training, answers_for_rating = _split_answers(all_answers, answers, score_type, training_size)
        _data_writer.add_line(f"Distribution in % of {source} answers for rating for question: {question_id}", _calculate_distribution(answers_for_rating, score_type))
        _data_writer.add_line(f"Distribution in % of {source} answers for training for question: {question_id}", _calculate_distribution(answers_for_training, score_type))
        _data_writer.add_line(f"Number of {source} answers for rating for question: {question_id}", len(answers_for_rating))
        _data_writer.add_line(f"Number of {source} answers for training for question: {question_id}", len(answers_for_training))
        _data_writer.add_line(f"Number of dropped {source} answers for question: {question_id}", len(all_answers) - len(answers_for_rating) - len(answers_for_training))
        write_answers_tsv(training_path, [answers_for_training], True)
        write_answers_tsv(rating_path, [answers_for_rating], True)


def _delete_file(file):
    # cleanup
    if os.path.exists(file):
        os.remove(file)


def _cleanup(config:Configuration, source:str):
    _delete_file(config.get_samples_for_training_path(source))
    _delete_file(config.get_samples_for_training_path("experts"))
    _delete_file(config.get_samples_for_testing_path(source))
    _delete_file(config.get_samples_for_testing_path("experts"))
    _delete_file(config.get_distribution_path())

def setup_args():
    parser = argparse.ArgumentParser(description='Randomly pick samples form a given source based on the distribution of the dataset created by human experts')
    parser.add_argument('source', help='The source of the data davinci, turbo, gpt4')

if __name__ == "__main__":
    config = Configuration()
    config_logger(log.DEBUG, "pick_random_samples.log")
    args = setup_args()

    ai_answers_per_question = get_answers_per_question(config.get_samples_path(args.source))
    # the distribution is always oriented on the experts dataset
    experts_answers_per_question = get_answers_per_question(config.get_samples_path("experts"))

    _cleanup(config)

    ai_answers_redistributed_per_question = {}
    experts_answers_redistributed_per_question = {}
    for question_id, experts_answers in experts_answers_per_question.items():
        ai_answers = ai_answers_per_question[question_id]

        # distribution for score_1 and score_2
        experts_answers_distribution = {score_type: _calculate_distribution(experts_answers, score_type) for score_type in range(1, 3)}
        log.debug(f"Calculated the following distributions: {experts_answers_distribution} for question: {question_id}")

        # find best fitting distribution
        all_ai_answers_redistributed = {f"{score_type_1}_{score_type_2}": _adjust_distribution(ai_answers, experts_answers_distribution[score_type_1], score_type_2) 
                                            for score_type_1 in range(1, 3) for score_type_2 in range(1, 3)}

        largest_distribution = max(all_ai_answers_redistributed, key=lambda k: len(all_ai_answers_redistributed[k]))

        log.debug(f"largest distribution: {largest_distribution}")
        log.debug(f"all_ai_answers_redistributed: {all_ai_answers_redistributed}")
        ai_answers_redistributed = all_ai_answers_redistributed[largest_distribution]
        redistributed_count = len(ai_answers_redistributed)
        original_count = len(ai_answers)

        if redistributed_count != original_count:
            log.debug(f"Updated Distribution in % of AI answers from: {original_count} to: {redistributed_count}")

        score_types = list(map(int, largest_distribution.split('_')))
        
        _data_writer.add_line(f"Ratio of used answers ai vs. experts for question: {question_id}", len(ai_answers_redistributed) / len(experts_answers))

        # we require at leased 3200 answers to train the model + 100 to rate
        ai_answers_redistributed = _adjust_distribution(ai_answers, experts_answers_distribution[score_types[0]], score_types[1], 3300)

        ai_answers_redistributed_per_question[question_id] = {"score_type": score_types[1], "answers": ai_answers_redistributed, "all_answers": ai_answers}
        experts_answers_redistributed_per_question[question_id] = {"score_type": score_types[0], "answers": experts_answers, "all_answers": experts_answers}

    _split_samples_for_question(ai_answers_redistributed_per_question, args.source, 3200, config)
    _split_samples_for_question(experts_answers_redistributed_per_question, "experts", 1600, config)
    _data_writer.write_to_file(config.get_distribution_path())