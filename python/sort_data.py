import csv
import os
from typing import Dict

def read_questions(file) -> Dict:
    questions = {}
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            # {'question_id': question}
            questions[row[0]] = row[1]
    
    return questions

def read_answers(file):
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        answers = [row for row in reader]
    return answers

def write_to_tsv(file, question_id, question, answers):
    mode = 'a' if os.path.exists(file) else 'w'
    with open(file, mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if mode == 'w':
            # Header
            writer.writerow(['Question', 'Answer', 'Score', 'IsQuestion', 'AnswerId', 'QuestionId'])

        writer.writerow([question, '', '', 'Y', '', question_id])
        for row in answers:
            writer.writerow(['', row[3], row[2], 'N', row[1], question_id])

if __name__ == "__main__":
    result_file = "./all_question_answers.tsv"
    questions = read_questions('./sorted_questions.tsv')
    for id, question in questions.items():
        answers_file = f"./SRA_numeric_allAnswers_prompt{id}.tsv"
        answers = read_answers(answers_file)
        write_to_tsv(result_file, id, question, answers)
        