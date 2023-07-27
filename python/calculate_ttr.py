import re
import numpy as np
from typing import List
from tsv_utils import Answer, get_answers_per_question

from config_logger import config_logger
from config import Configuration
import logging as log


def _calculate_ttrs(answers:List[Answer]):  
    ttrs = []
    all_words = []
    for answer in answers: 
        # tokenize the text by non-alphabetic characters
        words = re.findall(r'\b\w+\b', answer.answer.lower())
        ttrs.append(len(set(words))/len(words))
        all_words.extend(words)

    return np.mean(ttrs), len(set(all_words))/len(all_words)


if __name__ == "__main__":
    config = Configuration()
    config_logger(log.INFO, "calculate_ttr.log")

    ai_answers_per_question = get_answers_per_question(config.get_ai_answers_path())
    man_answers_per_question = get_answers_per_question(config.get_man_answers_path())
    
    for question, answers in ai_answers_per_question.items():
        average_ttr_per_answer, total_ttr = _calculate_ttrs(answers)
        log.info(f"TTR for question: {question} for AI: {total_ttr}")
        log.info(f"Average TTR per Answer for question: {question} for AI: {average_ttr_per_answer}")
    
    for question, answers in man_answers_per_question.items():
        average_ttr, ttrs = _calculate_ttrs(answers)
        log.info(f"TTR for question: {question} for MAN: {total_ttr}")
        log.info(f"Average TTRs per Answer for question: {question} for MAN: {average_ttr_per_answer}")