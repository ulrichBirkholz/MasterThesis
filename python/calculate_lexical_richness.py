import logging as log
from typing import List, Dict, Tuple, Any, Callable, Union

import numpy as np
import os
from tsv_utils import Answer, get_answers_per_question
from lexicalrichness import LexicalRichness

from config_logger import config_logger
from config import Configuration
import csv
import random
import shutil
import matplotlib.pyplot as plt


# Source: https://lexicalrichness.readthedocs.io/en/latest/#lexicalrichness.LexicalRichness.ttr
class LRCalculator:
    """ Helper class for calculating the lexical richness of answers

    This class uses various algorithms to measure the lexical richness or diversity of the given answers. It also 
    provides safeguards against problematic calculations or conditions that might cause errors or inaccuracies.

    Attributes:
        lex (LexicalRichness): An instance of the LexicalRichness class initialized with the input words
        methods (List[str]): A list of methods or algorithms to compute lexical diversity
        not_one (List[str]):  Methods that may divide by zero if the number of words is exactly one
        not_equal (List[str]): Methods that may divide by zero if the number of words equals the number of terms
        special (Dict[str, List[Dict[str, int]]]): Specific conditions where certain algorithms don't function correctly
    """
    # The order of _LABELS must match the order of _METHODS
    _LABELS = ["Number of Terms", "Number of Words", "Type-Token Ratio", "Root Type-Token Ratio (Guiraud's R)",
              "Corrected Type-Token Ratio", "Herdan’s C", "Summer", "Maas", "Dugast", "Yule's K", "Yule's I", "Herdan's VM", "Simpson's D"]

    _METHODS = ['terms', 'words', 'ttr', 'rttr', 'cttr', 'Herdan', 'Summer', 'Maas', 'Dugast', 'yulek', 'yulei', 'herdanvm', 'simpsond']


    @classmethod
    def LABELS(self) -> List[str]:
        return self._LABELS


    def __init__(self, words:str):
        """Initializes the LRCalculator with the given words."""
        self.lex = LexicalRichness(words)
        
        # divide for example by the logarithm of the word count
        self.not_one = ['Herdan', 'Summer', 'Maas', 'simpsond']

        # divide by zero if words equals terms
        self.not_equal = ['Dugast', 'yulei']

        # particular conditions that do not work for certain algorithms
        self.special = {
            "herdanvm": [{
                "words": 19,
                "terms": 19
            }, {
                "words": 21,
                "terms": 21
            }, {
                "words": 38,
                "terms": 38
            }]
        }


    def _safe_calculation(self, calculation:Callable[[], float], method:str) -> Union[float, Any]:
        """ Executes the given calculation while handling potential problematic scenarios

        Args:
            calculation (Callable[[], float]): The actual function or calculation to be executed
            method (str): The name of the method being executed to determine which scenarios to check for

        Returns:
            Union[float, Any]: Result of the calculation or np.nan if the value is problematic
        """
        log.info(f"Calculate method: {method}. Words: {self.lex.words}, Terms: {self.lex.terms}")

        # These methods divide by log(words)
        if method in self.not_one and self.lex.words == 1:
            log.info(f"Skip calculation - it can't be performed with only one word: {method}. Words: {self.lex.words}, Terms: {self.lex.terms}")
            return np.nan

        # These methods divide by words - terms
        if method in self.not_equal and self.lex.words == self.lex.terms:
            log.info(f"Skip calculation - it can't be performed if words and terms are equal in number: {method}. Words: {self.lex.words}, Terms: {self.lex.terms}")
            return np.nan

        # certain conditions that do not work
        if method in self.special:
            for condition in self.special[method]:
                if self.lex.words == condition["words"] and self.lex.terms == condition["terms"]:
                    log.info(f"Skip calculation - it can't be performed because of a special condition: {method}. Words: {self.lex.words}, Terms: {self.lex.terms}")
                    return np.nan

        try:
            return calculation()
        except Exception as e:
            log.error(f"Exception in _safe_calculation: {str(e)}, for method: {method}. Words: {self.lex.words}, Terms: {self.lex.terms}")
            return np.nan


    @property
    def data(self) -> Tuple[float, ...]:
        """ Computes lexical diversity for the text using all defined algorithms

        Returns:
            Tuple[float, ...]: Tuple of lexical diversity measurements from all methods
        """
        return tuple(self._safe_calculation(lambda: getattr(self.lex, method), method) for method in self._METHODS)


    @staticmethod
    def calculate_lr(answers:List[Answer]) -> Tuple[float, Any, Tuple[float, ...]]:
        """ Computes lexical richness for a list of answers both individually and as a whole

        Args:
            answers (List[Answer]): List of answers for which lexical diversity should be computed

        Returns:
            Tuple[float, Any, Tuple[float, ...]]: A tuple containing:
            - Average lexical diversity per answer
            - Count of answers not used in calculations due to specific conditions
            - Lexical diversity of the combined answers
        """
        average_lr = [LRCalculator(answer.answer).data for answer in answers]
        average_lr_mask = np.isnan(average_lr)

        return np.nanmean(np.array(average_lr), axis=0), np.sum(average_lr_mask, axis=0), LRCalculator(' '.join([answer.answer for answer in answers])).data

# TODO: documentation
class LRFigures:
    _COLORS = ['r', 'b', 'g', 'c', 'm']
    _COLOR_NAMES = ['red', 'blue', 'green', 'cyan', 'magenta']
    def __init__(self, data_source:str) -> None:
        self.data_source = data_source
        self.metrics = {}


    def plot(self, metrics:List[str], number_of_answers:str, values:List[float], question_id:str) -> None:

        for index, metric in enumerate(metrics):
            if not metric in self.metrics:
                self.metrics[metric] = {}

            if question_id in self.metrics[metric]:
                self.metrics[metric][question_id]['labels'].append(number_of_answers)
                self.metrics[metric][question_id]['values'].append(values[index])
            else:
                self.metrics[metric][question_id] = {
                    "labels": [number_of_answers],
                    "values": [values[index]]
                }


    def save(self, config:Configuration) -> None:

        for metric_name, sub_dict in self.metrics.items():
            title = f"Lexical Diversity of samples generated by {self.data_source}\nThe displayed metric is {metric_name}"
        
            figure = plt.figure()
            plt.title(title)

            plt.xticks(rotation=45, ha='right')
            x_label = "Number of Answers\n"
            x_label += "_"*80
            
            i = 0
            log.debug(f"Plot metric: {metric_name}")
            for question_id, metric in sub_dict.items():
                color = self._COLORS[i]
                log.debug(f"Plot color: {color} for question: {question_id} with i: {i}")

                plt.plot(metric["labels"], metric["values"], f"{color}o")
                x_label += f"\nEssey Set {question_id} is represented as {self._COLOR_NAMES[i]}"
                i += 1

            plt.xlabel(x_label)
            plt.ylabel(metric_name)
            
            filename = f"lexical_richness_{self.data_source}_{metric_name}.pdf"
            log.info(f"Save Diagram: {filename}")

            figure.savefig(config.get_path_for_lr_file(filename), bbox_inches='tight')
            plt.close()


def _calculate_and_save_lexical_richness(path, answers_per_question:Dict[str, Answer], data_source:str, number_of_answers:Union[int, None], diagrams:LRFigures) -> None:
    """ Compute metrics of lexical richness/diversity and save them to a TSV file.

    The function calculates various lexical richness metrics for a set of answers and saves 
    the results in a TSV file. If the file already exists, the results are appended; otherwise, 
    a new file is created.

    Args:
        path (str): Path to the TSV file where the results should be saved
        answers_per_question (Dict[str, Answer]): Answers categorized by their respective questions
        data_source (str): The origin or source of the provided data
        number_of_answers (Union[int, None]): The number of answers to use, if this value is None, all answers will be used. Default is None.

    Notes:
        The lexical richness metrics include "Number of Terms", "Type-Token Ratio", "Root Type-Token Ratio", 
        and several others. The function computes both total and average values for these metrics.
    """
    labels = LRCalculator.LABELS()
    mode = 'a' if os.path.exists(path) else 'w'
    with open(path, mode, newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        if mode == 'w':
            writer.writerow(['Question', 'Number of Answers', 'Represents'] + labels + [label + ' Skipped' for label in labels])  # Header row

        for question, answers_for_question in answers_per_question.items():
            if number_of_answers:
                if len(answers_for_question) < number_of_answers:
                    return

                answers = _get_random_answers(answers_for_question, number_of_answers)
            else:
                answers = answers_for_question

            average_lr, skipped_answers, total_lr = LRCalculator.calculate_lr(answers)

            diagrams.plot(labels, len(answers), total_lr, question)
            # Total for all answers
            writer.writerow([question, len(answers), data_source + ' Total'] + list(total_lr) + ['N/A']*len(labels))

            # Average per answer
            writer.writerow([question, len(answers), data_source + ' Average'] + list(average_lr) + list(skipped_answers))


def _get_random_answers(answers:List[Answer], number:int) -> List[Answer]:
    """ Select a random Set of Answers from a given List

    Args:
        answers (List[Answer]): The List of Answers to select form
        number (int): The number of Answers to be selected

    Returns:
        List[Answer]: Randomly selected Answers
    """
    random.shuffle(answers)
    return answers[:number]


def _cleanup(config:Configuration) -> None:
    """ Ensures that none of the files, created by this module, already exist

    Args:
        config (Configuration): Allows access to the projects central configuration
    """
    base_folder = config.get_lr_root_path()
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)

    os.makedirs(base_folder, exist_ok=True)


if __name__ == "__main__":
    config = Configuration()
    config_logger(log.WARNING, "calculate_lexical_richness.log")
    lr_file_path = config.get_lr_calculations_path()

    _cleanup(config)

    davinci_answers_per_question = get_answers_per_question(config.get_samples_path("davinci"))
    gpt4_answers_per_question = get_answers_per_question(config.get_samples_path("gpt4"))
    man_answers_per_question = get_answers_per_question(config.get_samples_path("experts"))

    davinci_diagrams = LRFigures("davinci")
    gpt4_diagrams = LRFigures("gpt4")
    experts_diagrams = LRFigures("experts")
    for size in [None, 100, 200, 400, 800, 1200, 1600, 2400, 3200]:
        _calculate_and_save_lexical_richness(lr_file_path, davinci_answers_per_question, "text-davinci-003", size, davinci_diagrams)
        _calculate_and_save_lexical_richness(lr_file_path, gpt4_answers_per_question, "gpt4", size, gpt4_diagrams)
        _calculate_and_save_lexical_richness(lr_file_path, man_answers_per_question, "experts", size, experts_diagrams)
    
    davinci_diagrams.save(config)
    gpt4_diagrams.save(config)
    experts_diagrams.save(config)

