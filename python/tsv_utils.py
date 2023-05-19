import csv
import os
import logging as log

# TODO: refactor, isolate duplicated code and such

# Parse a given tsv file and use SampleAnswer depending on the property 'sample'
# Question | SampleAnswer (Optional) | QuestionId
def get_questions(file, use_sample):
    questions = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            parsed_row = []
            log.debug(f'Parse Question: {row[0]}')
            parsed_row.append(row[0])
            if use_sample and len(row) > 1:
                log.debug(f'Parse SampleAnswer: {row[1]}')
                parsed_row.append(row[1])
            else:
                log.debug("Skip SampleAnswer")
                parsed_row.append(None)
            log.debug(f'Parse QuestionId: {row[-1]}')
            parsed_row.append(row[-1])
            questions.append(parsed_row)
    return questions

# {"question_id":[{QuestionId | Answer | AnswerId | Score1 | Score2}...]}
# Score 1 is just for reference and can be ignored here
def get_answers_per_question(file):
    answers = {}
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            log.debug(f'Parse Answer: {row[2]}')
            formatted_row = {
                "question_id":row[0],
                "answer":row[1],
                "answer_id":row[2],
                "score": row[4]
            }
            if row[0] in answers:
                answers[row[0]].append(formatted_row)
            else:
                answers[row[0]] = [formatted_row]
    return answers

# {"question_id":[{QuestionId | Answer | AnswerId}...]}
# Score 1 is just for reference and can be ignored here
def get_answers_to_rate_per_question(file):
    answers = {}
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            log.debug(f'Parse Answer: {row[2]}')
            formatted_row = {
                "question_id":row[0],
                "answer":row[1],
                "answer_id":row[2]
            }
            if row[0] in answers:
                answers[row[0]].append(formatted_row)
            else:
                answers[row[0]] = [formatted_row]
    return answers

# Question and answer are explicitly not stored here -> redundant information
# Instead we will setup a python script which exports them out in a nicely formatted way
# QuestionId | AnswerId | Score1 | Score2
def write_rated_answers_tsv(file, answers, extend):
    mode = 'a' if extend and os.path.exists(file) else 'w'

    log.debug(f"Writing rated answers: {answers} to file: {file}")
    with open(file, mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if mode is 'w':
            writer.writerow(['QuestionId', 'AnswerId', 'Score1', 'Score2'])
        for row in answers:
            writer.writerow(row)

# TODO: decide with what we want to gor and remove the other method
# QuestionId | Answer | AnswerId | Score
def write_answers_tsv(file, answers, extend):
    mode = 'a' if extend and os.path.exists(file) else 'w'

    log.debug(f"Writing answers: {answers} to file: {file}")
    with open(file, mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if mode is 'w':
            writer.writerow(['QuestionId', 'Answer', 'AnswerId', 'Score'])
        for row in answers:
            writer.writerow(row)

# QuestionId | Answer | AnswerId | Score1 | Score2
def write_answers_tsv_2(file, answers, extend):
    mode = 'a' if extend and os.path.exists(file) else 'w'

    log.debug(f"Writing answers: {answers} to file: {file}")
    with open(file, mode, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if mode is 'w':
            writer.writerow(['QuestionId', 'Answer', 'AnswerId', 'Score1', 'Score2'])
        for row in answers:
            writer.writerow(row)

##################################