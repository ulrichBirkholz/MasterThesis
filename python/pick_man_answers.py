import csv
from config import Configuration
import os
from config_logger import config_logger
import logging as log

if __name__ == "__main__":
    config_logger(log.DEBUG, 'pick_man_answers.log')
    config = Configuration()

    target_file = config.get_man_answers_path()
    src_file = config.get_man_answers_src_path()

    if os.path.exists(target_file):
        os.remove(target_file)

    # From: Id (answer_id)    EssaySet (question_id)    Score1    Score2    EssayText (Answer)
    # To:    QuestionId    Answer    AnswerId    Score1    Score2
    with open(src_file, 'r') as input_file, open(target_file, 'w') as output_file:
        tsv_reader = csv.reader(input_file, delimiter='\t')
        tsv_writer = csv.writer(output_file, delimiter='\t')

        tsv_writer.writerow(["QuestionId", "Answer", "AnswerId", "Score1", "Score2"])
        for row in tsv_reader:
            # We target question 5 and 6
            if row[1] in ["5","6"]:
                tsv_writer.writerow([row[1], row[4], row[0], row[2], row[3]])
            else:
                log.debug(f"Do not use Answer: {row[1]}")
