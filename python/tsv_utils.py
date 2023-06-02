import csv
import os
import logging as log
from dataclasses import dataclass
from typing import List, Any, Dict, Optional

# TODO: refactor, isolate duplicated code, strate up naming (answer, sample, rating score ect.) and such
# TODO: refactor, consumer must not require knowledge about the tsv files

# Classes
@dataclass
class Question:
    question: Optional[str] = None
    sample_answer: Optional[str] = None
    question_id: Optional[str] = None

@dataclass
class Answer:
    question_id: Optional[str] = None
    answer: Optional[str] = None
    answer_id: Optional[str] = None
    score_1: Optional[int] = None
    score_2: Optional[int] = None

@dataclass
class Rating:
    question: Optional[Question] = None
    answer: Optional[Answer] = None

# Load tsv files

# Parse a given tsv file and use SampleAnswer depending on the property 'sample'
# Question | SampleAnswer (Optional) | QuestionId
def get_questions(file:str, use_sample:bool) -> List[Question]:
    questions = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            log.debug(f'Parse Question: {row[0]}')
            parsed_row = Question(row[0])
            if use_sample and len(row) > 1:
                log.debug(f'Parse SampleAnswer: {row[1]}')
                parsed_row.sample_answer = row[1]

            log.debug(f'Parse QuestionId: {row[-1]}')
            parsed_row.question_id = row[-1]
            questions.append(parsed_row)
    return questions

# {"question_id": [Answer]}
# Score 1 is just for reference
def get_answers_per_question(file:str) -> Dict:
    answers = {}
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            log.debug(f'Parse Answer: {row[2]}')
            answer = Answer(row[0], row[1], row[2])

            # Add score if applicable
            if len(row) == 5:
                answer.score_1 = row[3]
                answer.score_2 = row[4]

            # Sort answers per question_id
            if answer.question_id in answers:
                answers[answer.question_id].append(answer)
            else:
                answers[answer.question_id] = [answer]

    return answers

# {"question_id": [Question]}
def _get_questions_per_question_id(file:str) -> Dict:
    question_list = get_questions(file, False)
    questions = {question.question_id: question.question for question in question_list}
    return questions

def get_ratings(rated_file:str, questions_file:str) -> List[Rating]:
    ratings = []
    questions = _get_questions_per_question_id(questions_file)

    # QuestionId | AnswerId | Score1 | Score2
    with open(rated_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            question = questions[row[0]]
            # NOTE: if we need the answer text or the original annotation use: get_answers_per_question
            rating = Rating(question, Answer(row[0], None, row[1], row[2], None))
            if len(row) == 4:
                rating.answer.score_2 = row[3]
            ratings.append(rating)
    return ratings

# Save tsv files

# Question and answer are explicitly not stored here -> redundant information
# Instead we will setup a python script which exports them out in a nicely formatted way
# Score1 is generated by the trained model, Score2 represents the reference rating from ChatGTP
# QuestionId | AnswerId | Score1 | Score2
def write_rated_answers_tsv(file: str, answers: List[Answer], extend:bool) -> None:
    mode = 'a' if extend and os.path.exists(file) else 'w'

    log.debug(f"Writing rated answers: {answers} to file: {file}")
    with open(file, mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if mode == 'w':
            writer.writerow(['QuestionId', 'AnswerId', 'Score1', 'Score2'])
        for answer in answers:
            writer.writerow([answer.question_id, answer.answer_id, answer.score_1, answer.score_2])

# Score1 is generated along with the answer, Score2 is separately assigned
# QuestionId | Answer | AnswerId | Score1 | Score2
def write_answers_tsv_2(file:str, answers:List[Answer], extend:bool) -> None:
    mode = 'a' if extend and os.path.exists(file) else 'w'

    log.debug(f"Writing answers: {answers} to file: {file}")
    with open(file, mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if mode == 'w':
            writer.writerow(['QuestionId', 'Answer', 'AnswerId', 'Score1', 'Score2'])
        for row in answers:
            writer.writerow(_to_array(row))

# utility
def _to_array(obj: Any) -> List:
    return [getattr(obj, field) for field in obj.__dataclass_fields__.keys()]