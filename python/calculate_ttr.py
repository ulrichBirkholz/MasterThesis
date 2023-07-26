import re
import numpy as np
from typing import List
from tsv_utils import Answer, get_answers_per_question

from config_logger import config_logger
from config import Configuration
import logging as log


# calculate average ttr for the given answers
def _calculate_ttrs(answers:List[Answer]):
    
    ttrs = []
    for answer in answers: 
        # tokenize the text by non-alphabetic characters
        words = re.findall(r'\b\w+\b', answer.answer.lower())
        ttrs.append(len(set(words))/len(words))
    
    return np.mean(ttrs), ttrs


if __name__ == "__main__":
    config = Configuration()
    config_logger(log.INFO, "calculate_ttr.log")

    ai_answers_per_question = get_answers_per_question(config.get_ai_answers_path())
    man_answers_per_question = get_answers_per_question(config.get_man_answers_path())
    
    for question, answers in ai_answers_per_question.items():
        average_ttr, ttrs = _calculate_ttrs(answers)
        log.info(f"Average ttrs for question: {question} for AI: {average_ttr}")
    
    for question, answers in man_answers_per_question.items():
        average_ttr, ttrs = _calculate_ttrs(answers)
        log.info(f"Average ttrs for question: {question} for MAN: {average_ttr}")