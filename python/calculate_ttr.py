import logging as log
from typing import List, Dict

import numpy as np
import os
import time
from tsv_utils import Answer, get_answers_per_question
from lexicalrichness import LexicalRichness

from config_logger import config_logger
from config import Configuration
import csv


# Source: https://lexicalrichness.readthedocs.io/en/latest/#lexicalrichness.LexicalRichness.ttr
class TTRCalculator:
    def __init__(self, words):
        self.lex = LexicalRichness(words)
        self.methods = ['terms', 'words', 'ttr', 'rttr', 'cttr', 'Herdan', 'Summer', 'Maas', 'Dugast', 'yulek', 'yulei', 'herdanvm', 'simpsond']
        
        # use for example a logarithm of the word count
        self.not_one = ['Herdan', 'Summer', 'Maas', 'simpsond']

        # divide by zero if words equals terms
        self.not_equal = ['Dugast', 'yulei']

        # certain conditions that do not work
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


    def _safe_calculation(self, calculation, method):
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
    def data(self):
        return tuple(self._safe_calculation(lambda: getattr(self.lex, method), method) for method in self.methods)

    @staticmethod
    def calculate_ttrs(answers: List[Answer]):
        average_ttrs = [TTRCalculator(answer.answer).data for answer in answers]
        average_ttrs_mask = np.isnan(average_ttrs)

        return np.nanmean(np.array(average_ttrs), axis=0), np.sum(average_ttrs_mask, axis=0), TTRCalculator(' '.join([answer.answer for answer in answers])).data

def _write_ttrs_to_tsv(path, answers_per_question: Dict[str, Answer], answer_type: str):
    labels = ["Number of Terms", "Number of Words", "Type-Token Ratio", "Root Type-Token Ratio (Guiraud's R)",
              "Corrected Type-Token Ratio", "Herdan’s C", "Summer", "Maas", "Dugast", "Yule's K", "Yule's I", "Herdan's VM", "Simpson's D"]

    mode = 'a' if os.path.exists(path) else 'w'
    with open(path, mode, newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        if mode == 'w':
            writer.writerow(['Question', 'Number of Answers', 'Represents'] + labels + [label + ' Skipped' for label in labels])  # Header row

        for question, answers in answers_per_question.items():
            average_ttr, skipped_answers, total_ttr = TTRCalculator.calculate_ttrs(answers)

            # Total for all answers
            writer.writerow([question, len(answers), answer_type + ' Total'] + list(total_ttr) + ['N/A']*len(labels))

            # Average per answer
            writer.writerow([question, len(answers), answer_type + ' Average'] + list(average_ttr) + list(skipped_answers))


if __name__ == "__main__":
    config = Configuration()
    config_logger(log.WARNING, "calculate_ttr.log")
    ttr_file_path = config.get_ttr_calculations_path()

    # cleanup
    if os.path.exists(ttr_file_path):
        os.remove(ttr_file_path)

    ai_answers_per_question = get_answers_per_question(config.get_ai_answers_path())
    man_answers_per_question = get_answers_per_question(config.get_man_answers_path())
    
    _write_ttrs_to_tsv(ttr_file_path, ai_answers_per_question, "AI")
    _write_ttrs_to_tsv(ttr_file_path, man_answers_per_question, "Manual")
