import random
from tsv_utils import get_answers_per_question, write_answers_tsv, Answer
from config import Configuration
from collections import Counter
from typing import List, Dict, Tuple, Union
import logging as log
from config_logger import config_logger
import os
import argparse
from argparse import Namespace


class DataWriter:
    """
    A helper class to record and write data distributions to a file.
    
    This class facilitates the collection and organization of distribution data, 
    saving the gathered information to a designated file.

    Attributes:
        lines (List[str]): Accumulates the data distribution details before writing to the file.
    """

    def __init__(self):
        """Initializes the DataWriter with an empty list of lines."""
        self.lines = []

    def add_line(self, description: str, information: str) -> None:
        """
        Appends a new data distribution line to the internal list.
        
        Args:
            description (str): Describes the type or nature of the distribution data.
            information (str): Contains the specific details of the distribution.
        """
        self.lines.append(f"{description}: {information}")

    def write_to_file(self, path: str) -> None:
        """
        Writes all accumulated distribution data to a specified file.
        
        The lines containing the distribution data are sorted alphabetically before writing 
        to ensure a structured output.
        
        Args:
            path (str): The path to the file where the distribution data will be saved.
        """
        self.lines.sort()
        with open(path, 'a') as file:
            file.write('\n'.join(self.lines))


_data_writer = DataWriter()

# {'category', percent} -> {'0': 10, '1': 50, '2': 15, '3': 25}
def _calculate_distribution(samples:List[Answer], score_type:int, total:bool = False) -> Dict[str, int]:
    """ Calculates the distribution of categories (scores) of the given samples

    Args:
        samples (List[Answer]): Samples of which the distribution is calculated
        score_type (int): The score type (either 1 or 2) of which the distribution is calculated
        total (bool, optional): False to calculate the distribution in percent, otherwise True. Defaults to False.

    Returns:
        Dict[str, int]: The calculated distribution
    """
    category_counts = Counter([getattr(answer, f'score_{score_type}') for answer in samples])
    if total:
        return {category: category_counts[category] for category in sorted(category_counts)}
    else:
        total_count = len(samples)
        return {category: round((category_counts[category] / total_count) * 100) for category in sorted(category_counts)}


def _adjust_distribution(samples:List[Answer], target_distribution:Dict[str, int], score_type:int, min_size:int=0) -> List[Answer]:
    """ Refines the category distribution of a given sample list. The algorithm continues to modify the distribution 
    until it meets the specified minimum sample count.

    Args:
        samples (List[Answer]): The source list of samples to chose from
        target_distribution (Dict[str, int]): The desired target distribution
        score_type (int): The score type (either 1 or 2) of which the distribution is calculated
        min_size (int, optional): The minimal number of samples contained within the result List. Defaults to 0.

    Returns:
        List[Answer]: Redistributed List of Answers
    """
    total_count = len(samples)
    assert min_size < total_count, f"The minimal required result size: {min_size} is larger than the total number of answers: {total_count}"

    category_items = {category: [answer for answer in samples if getattr(answer, f'score_{score_type}') == category] for category in target_distribution}

    lowest_category_ratio = min(len(category_items[category]) / round((target_percent / 100) * total_count) 
                                for category, target_percent in target_distribution.items())

    log.debug(f"Lowest category ratio: {lowest_category_ratio}")

    result = []
    tolerance = 0
    # increase tolerance until we reach the desired number of answers
    while (len(result) < min_size):
        result.clear()
        for category, target_percent in target_distribution.items():
            number_of_samples = len(category_items[category])
            number_to_pick = round((target_percent / 100) * total_count * lowest_category_ratio) + tolerance

            if number_to_pick > number_of_samples:
                number_to_pick = number_of_samples

            result.extend(random.sample(category_items[category], number_to_pick))
        tolerance += 1

    return result


def _select_answers_for_category(category_item:List[Answer], factor:float) -> List[Answer]:
    """ Picks a number of random Answers from a given category depending on a given factor

    Args:
        category_item (List[Answer]): Answers to chose form
        factor (float): Factor of answers to chose (0 < factor < 1)

    Returns:
        List[Answer]: Random selection of Answers
    """
    category_len = len(category_item)
    sample_size = round(category_len * factor)
    
    log.debug(f"About to pick {sample_size} random answers form {category_len} samples")
    result = random.sample(category_item, sample_size)

    for answer in result:
        category_item.remove(answer)

    return result


def _split_answers(all_answers:List[Answer], distributed_answers:List[Answer], score_type:int, training_size:int) -> Tuple[List[Answer], List[Answer]]:
    """ Splits answers into training and testing datasets based on the provided distribution and specified training size.
    
    The function creates a training dataset using the desired distribution of categories from `distributed_answers`. 
    All remaining samples from `all_answers` that aren't part of the training dataset are added to the testing dataset.

    Args:
        all_answers (List[Answer]): All available samples
        distributed_answers (List[Answer]): A list of answers reflecting the desired distribution of categories
        score_type (int): The score type (either 1 or 2) of which the distribution is calculated
        training_size (int): The desired number of samples in the training dataset

    Returns:
        Tuple[List[Answer], List[Answer]]: A tuple containing two lists - the training dataset and the testing dataset
    """    

    assert len(distributed_answers) >= training_size, f"""The distribution is too small!
We need at leased {training_size} answers for training the models, but we have only {len(distributed_answers)}"""

    assert len(all_answers) >= training_size + 100, f"""The distribution is too small!
We need at leased {training_size} answers for training + 100 answers for testing the models, but we have only {len(distributed_answers)}"""

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
    
    answers_for_testing = [answer for answer in all_answers if answer not in answers_for_training]
    return answers_for_training, answers_for_testing


def _split_samples_for_question(sample_selection_info_per_question:Dict[str, Dict[str, Union[int, List[Answer]]]], data_source:str, training_size:int, config:Configuration) -> None:
    """ Divides samples for each question into training and testing datasets, determined by the specified training_size
    and the total number of available samples.

    Args:
        sample_selection_info_per_question (Dict[str, Dict[str, Union[int, List[Answer]]]]): All relevant information to perform the selection of samples for each question
        data_source (str): The source of generated data.
        training_size (int): The desired number of samples in the training dataset
        config (Configuration): Allows access to the projects central configuration
    """
    training_path = config.get_samples_for_training_path(data_source)
    rating_path = config.get_samples_for_testing_path(data_source)

    for question_id, distribution in sample_selection_info_per_question.items():
        log.debug(f"Split {data_source} answers for question: {question_id}")

        score_type = distribution["score_type"]
        answers = distribution["answers"]

        all_answers = distribution["all_answers"]
        _data_writer.add_line(f"{data_source} answers for question: {question_id} use score_type", score_type)
        _data_writer.add_line(f"Distribution in % of all {data_source} answers for question: {question_id}", _calculate_distribution(all_answers, score_type))
        _data_writer.add_line(f"Distribution of all {data_source} answers for question: {question_id}", _calculate_distribution(all_answers, score_type, True))

        _data_writer.add_line(f"Distribution in % of used {data_source} answers for question: {question_id}", _calculate_distribution(answers, score_type))
        _data_writer.add_line(f"Distribution of used {data_source} answers for question: {question_id}", _calculate_distribution(answers, score_type, True))

        _data_writer.add_line(f"Number of all {data_source} answers for question: {question_id}", len(all_answers))
        log.debug(f"distribution contains: {len(answers)} answers")

        answers_for_training, answers_for_testing = _split_answers(all_answers, answers, score_type, training_size)
        _data_writer.add_line(f"Distribution in % of {data_source} answers for rating for question: {question_id}", _calculate_distribution(answers_for_testing, score_type))
        _data_writer.add_line(f"Distribution in % of {data_source} answers for training for question: {question_id}", _calculate_distribution(answers_for_training, score_type))
        _data_writer.add_line(f"Number of {data_source} answers for rating for question: {question_id}", len(answers_for_testing))
        _data_writer.add_line(f"Number of {data_source} answers for training for question: {question_id}", len(answers_for_training))
        _data_writer.add_line(f"Number of dropped {data_source} answers for question: {question_id}", len(all_answers) - len(answers_for_testing) - len(answers_for_training))
        write_answers_tsv(training_path, [answers_for_training], True)
        write_answers_tsv(rating_path, [answers_for_testing], True)


def _delete_file(file:str) -> None:
    """ Deletes a given file if it exists

    Args:
        file (str): Path to the file
    """
    # cleanup
    if os.path.exists(file):
        os.remove(file)


def _cleanup(config:Configuration, args:Namespace) -> None:
    """ Ensures that none of the files, created by this module, already exist

    Args:
        config (Configuration): Allows access to the projects central configuration
        args (Namespace): Execution Arguments
    """
    if args.davinci:
        _delete_file(config.get_samples_for_training_path("davinci"))
        _delete_file(config.get_samples_for_testing_path("davinci"))
    
    if args.turbo:
        _delete_file(config.get_samples_for_training_path("turbo"))
        _delete_file(config.get_samples_for_testing_path("turbo"))
    
    if args.gpt4:
        _delete_file(config.get_samples_for_training_path("gpt4"))
        _delete_file(config.get_samples_for_testing_path("gpt4"))

    _delete_file(config.get_samples_for_training_path("experts"))
    _delete_file(config.get_samples_for_testing_path("experts"))
    _delete_file(config.get_distribution_path())

# example: python -m pick_random_samples --davinci --turbo --gpt4
def setup_args() -> Namespace:
    """ Setup of the execution arguments

    Returns:
        Namespace: arguments to be used
    """
    parser = argparse.ArgumentParser(description='Randomly pick samples form a given datasource based on the distribution of the dataset created by human experts')
    parser.add_argument('--davinci', action='store_true', help='Include samples created by text-davinci-003')
    parser.add_argument('--turbo', action='store_true', help='Include samples annotated by gpt-3.5-turbo')
    parser.add_argument('--gpt4', action='store_true', help='Include samples created by gpt4')
    return parser.parse_args()


def _pick_samples_for(ai_data_source:str, experts_answers_per_question:Dict[str, List[Answer]], config:Configuration) -> None:
    """ Splits samples into two datasets: training and testing.

    This function aims to create a training dataset from both the provided AI data source and expert-generated samples, 
    ensuring the category distribution is as consistent as possible between them.
    
    Note:
        The training dataset from the AI data source will contain 3200 samples.
        The training dataset from the experts will consist of 1600 samples.

    Args:
        ai_data_source (str): The source of AI-generated data.
        experts_answers_per_question (Dict[str, List[Answer]]): A dictionary containing experts' answers per question.
        config (Configuration): Allows access to the projects central configuration
    """

    ai_answers_per_question = get_answers_per_question(config.get_samples_path(ai_data_source))

    ai_selection_info_per_question = {}
    experts_selection_info_per_question = {}
    for question_id, experts_answers in experts_answers_per_question.items():
        ai_answers = ai_answers_per_question[question_id]

        # distribution for score_1 and score_2
        experts_answers_distribution = {score_type: _calculate_distribution(experts_answers, score_type) for score_type in [1, 2]}
        log.debug(f"Calculated the following distributions: {experts_answers_distribution} for question: {question_id}")

        # find best fitting distribution, which is the one containing the most samples
        all_ai_answers_redistributed = {f"{score_type_1}_{score_type_2}": _adjust_distribution(ai_answers, experts_answers_distribution[score_type_1], score_type_2) 
                                            for score_type_1 in [1, 2] for score_type_2 in [1, 2]}

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

        # we require at leased 3200 answers to train the model
        ai_answers_redistributed = _adjust_distribution(ai_answers, experts_answers_distribution[score_types[0]], score_types[1], 3200)

        ai_selection_info_per_question[question_id] = {"score_type": score_types[1], "answers": ai_answers_redistributed, "all_answers": ai_answers}
        experts_selection_info_per_question[question_id] = {"score_type": score_types[0], "answers": experts_answers, "all_answers": experts_answers}

    # All AI sources have up to 3200 samples available to train other models such as BERT and XG-Boost
    _split_samples_for_question(ai_selection_info_per_question, ai_data_source, 3200, config)
    _split_samples_for_question(experts_selection_info_per_question, "experts", 1600, config)


if __name__ == "__main__":
    config = Configuration()
    config_logger(log.DEBUG, "pick_random_samples.log")
    args = setup_args()

    _cleanup(config, args)

    # the distribution is always oriented on the experts dataset
    experts_answers_per_question = get_answers_per_question(config.get_samples_path("experts"))

    if args.davinci:
        _pick_samples_for("davinci", experts_answers_per_question, config)

    if args.turbo:
        _pick_samples_for("turbo", experts_answers_per_question, config)

    if args.gpt4:
        _pick_samples_for("gpt4", experts_answers_per_question, config)

    _data_writer.write_to_file(config.get_distribution_path())