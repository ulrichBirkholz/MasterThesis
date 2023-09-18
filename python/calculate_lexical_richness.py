import logging as log
from typing import List, Dict, Tuple, Any, Callable, Union

import numpy as np
import os
from tsv_utils import Answer, get_answers_per_question
from lexicalrichness import LexicalRichness

from config_logger import config_logger
from config import Configuration
import csv


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
    def __init__(self, words:str):
        """Initializes the LRCalculator with the given words."""
        self.lex = LexicalRichness(words)
        self.methods = ['terms', 'words', 'ttr', 'rttr', 'cttr', 'Herdan', 'Summer', 'Maas', 'Dugast', 'yulek', 'yulei', 'herdanvm', 'simpsond']
        
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


    def _safe_calculation(self, calculation:Callable[[], float], method:str) -> Union[float, np.nan]:   
        """ Executes the given calculation while handling potential problematic scenarios

        Args:
            calculation (Callable[[], float]): The actual function or calculation to be executed
            method (str): The name of the method being executed to determine which scenarios to check for

        Returns:
            Union[float, np.nan]: Result of the calculation or np.nan if the value is problematic
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
        return tuple(self._safe_calculation(lambda: getattr(self.lex, method), method) for method in self.methods)


    @staticmethod
    def calculate_ttrs(answers:List[Answer]) -> Tuple[float, Any, Tuple[float, ...]]:
        """ Computes lexical diversity for a list of answers both individually and as a whole

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


def _calculate_and_save_lexical_richness(path, answers_per_question:Dict[str, Answer], data_source:str) -> None:
    """ Compute metrics of lexical richness/diversity and save them to a TSV file.

    The function calculates various lexical richness metrics for a set of answers and saves 
    the results in a TSV file. If the file already exists, the results are appended; otherwise, 
    a new file is created.

    Args:
        path (str): Path to the TSV file where the results should be saved
        answers_per_question (Dict[str, Answer]): Answers categorized by their respective questions
        data_source (str): The origin or source of the provided data

    Notes:
        The lexical richness metrics include "Number of Terms", "Type-Token Ratio", "Root Type-Token Ratio", 
        and several others. The function computes both total and average values for these metrics.
    """
    labels = ["Number of Terms", "Number of Words", "Type-Token Ratio", "Root Type-Token Ratio (Guiraud's R)",
              "Corrected Type-Token Ratio", "Herdan’s C", "Summer", "Maas", "Dugast", "Yule's K", "Yule's I", "Herdan's VM", "Simpson's D"]

    mode = 'a' if os.path.exists(path) else 'w'
    with open(path, mode, newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        if mode == 'w':
            writer.writerow(['Question', 'Number of Answers', 'Represents'] + labels + [label + ' Skipped' for label in labels])  # Header row

        for question, answers in answers_per_question.items():
            average_lr, skipped_answers, total_ttr = LRCalculator.calculate_ttrs(answers)

            # Total for all answers
            writer.writerow([question, len(answers), data_source + ' Total'] + list(total_ttr) + ['N/A']*len(labels))

            # Average per answer
            writer.writerow([question, len(answers), data_source + ' Average'] + list(average_lr) + list(skipped_answers))


if __name__ == "__main__":
    config = Configuration()
    config_logger(log.WARNING, "calculate_lexical_richness.log")
    lr_file_path = config.get_lr_calculations_path()

    # cleanup
    if os.path.exists(lr_file_path):
        os.remove(lr_file_path)

    davinci_answers_per_question = get_answers_per_question(config.get_samples_path("davinci"))
    gpt4_answers_per_question = get_answers_per_question(config.get_samples_path("gpt4"))
    man_answers_per_question = get_answers_per_question(config.get_samples_path("experts"))
    
    _calculate_and_save_lexical_richness(lr_file_path, davinci_answers_per_question, "text-davinci-003")
    _calculate_and_save_lexical_richness(lr_file_path, gpt4_answers_per_question, "gpt4")
    _calculate_and_save_lexical_richness(lr_file_path, man_answers_per_question, "experts")
