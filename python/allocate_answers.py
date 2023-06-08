import csv
import os
import logging as log
from config import Configuration
from dataclasses import dataclass
from typing import List, Any, Dict, Optional
from tsv_utils import write_answers_tsv_2, write_rated_answers_tsv
import random

@dataclass
class Answer:
    question_id: Optional[str] = None
    answer: Optional[str] = None
    answer_id: Optional[str] = None
    score_1: Optional[int] = None
    score_2: Optional[int] = None

def get_question_ids(file:str) -> List[str]:
    question_ids = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            log.debug(f'Parse QuestionId: {row[-1]}')
            question_ids.append(row[-1])
    return question_ids

def get_answers(file: str, question_id: str):
    answers = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            assert question_id == row[1], f"QuestionId mismatch in file {file}"
            answers.append(Answer(question_id, row[3], row[0], None, row[2]))
    return answers

def sort_answers(question_id:str):
    file = f"../SRA_numeric/SRA_numeric_allAnswers_prompt{question_id}.tsv"
    if not os.path.exists(file):
        print(f"file: {file} does not exist")
        return
    all_answers = get_answers(file, question_id)
    random_answer = random.choice(all_answers)
    answers_for_training = all_answers.copy()
    answers_for_training.remove(random_answer)

    write_answers_tsv_2(config.get_man_answers_path(), answers_for_training, True)
    write_answers_tsv_2(config.get_answers_to_rate_path(), [random_answer], True)
    
    

if __name__ == "__main__":
    config = Configuration()

    question_ids = get_question_ids(config.get_questions_path())
    #list(map(print, question_ids))
    list(map(sort_answers, question_ids))
        
            
            