import csv
import logging as log

# Parse a given tsv file and use SampleAnswer depending on the property 'sample'
# Question | SampleAnswer (Optional) | QuestionId
def get_questions(file, sample):
    questions = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            parsed_row = []
            log.debug(f'Parse Question: {row[0]}')
            parsed_row.append(row[0])
            if sample and len(row) > 1:
                log.debug(f'Parse SampleAnswer: {row[1]}')
                parsed_row.append(row[1])
            else:
                log.debug("Skip SampleAnswer")
                parsed_row.append(None)
            log.debug(f'Parse QuestionId: {row[-1]}')
            parsed_row.append(row[-1])
            questions.append(parsed_row)
    return questions

# QuestionId | Answer | AnswerId | Score
def write_answers_tsv(file, answers):
    log.debug(f"Writing answers: {answers} to file: {file}")
    print(f"Writing answers: {answers} to file: {file}")
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['QuestionId', 'Answer', 'AnswerId', 'Score'])
        for row in answers:
            writer.writerow(row)
