import csv
import os
import logging as log
from dataclasses import dataclass
from typing import List, Iterator, Any, Dict, Optional


@dataclass
class Question:
    """ Data representation of a question and its related attributes

    This class defines a structured representation for a question, 
    including an optional sample answer, a unique identifier, and a potential score offset

    Attributes:
        question (Optional[str]): The content of the question
        sample_answer (Optional[str]): A possible answer to the question, used for reference or guidance
        question_id (Optional[str]): A unique identifier for the question
        score_offset (Optional[int]): An offset value that can be applied to the scoring mechanism of the question
    """
    question: Optional[str] = None
    sample_answer: Optional[str] = None
    question_id: Optional[str] = None
    score_offset: Optional[int] = None


@dataclass
class Answer:
    """ Data representation of an answer and its associated attributes

    This class provides a structured format for an answer to a particular question, 
    complete with a unique identifier and optional scoring details

    Attributes:
        question_id (Optional[str]): The unique identifier for the question to which this answer corresponds
        answer (Optional[str]): The content or text of the answer
        answer_id (Optional[str]): A unique identifier for this specific answer
        score_1 (Optional[int]): A score associated with this answer
        score_2 (Optional[int]): Another score, distinct from score_1, related to this answer
    """
    question_id: Optional[str] = None
    answer: Optional[str] = None
    answer_id: Optional[str] = None
    score_1: Optional[int] = None
    score_2: Optional[int] = None


@dataclass
class KeyElement:
    """ Data representation of a key element associated with a specific ASAP Essay Set

    This class defines a structured representation for an essential aspect or component 
    linked to an essay prompt in the ASAP dataset

    Attributes:
        question_id (Optional[str]): The unique identifier for the essay prompt with which this key element is associated
        element (Optional[str]): The actual content or description of the key element
    """
    question_id: Optional[str] = None
    element: Optional[str] = None


# Load tsv files


def get_key_elements_by_question_id(file:str) -> Dict[str, List[KeyElement]]:
    """ Retrieve and group key elements from a given TSV file based on their associated questions

    This function reads a TSV file containing key elements. Each key element
    is associated with a specific question Id. The function then groups these key elements by 
    their respective question Ids and returns a dictionary representation

    Args:
        file (str): Path to the TSV file containing key elements

    Returns:
        Dict[str, List[KeyElement]]: A dictionary where keys are question Ids and values are 
                                    lists of associated key elements

    Notes:
        - The TSV file should have a header that this function will skip
        - The expected format for the columns in the file is: QuestionId | Element
    """
    key_elements = {}
    with open(file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            if len(row) == 0:
                log.warning(f"Found empty row in file: {file}")
                continue

            log.debug(f"Parse Key Element: {row}")
            element = KeyElement(row[0], row[1])

            # Sort answers per question_id
            if element.question_id in key_elements:
                key_elements[element.question_id].append(element)
            else:
                key_elements[element.question_id] = [element]

    return key_elements


def get_questions(file:str, use_sample:bool) -> List[Question]:
    """ Load and parse questions from a specified TSV file

    Given a file path pointing to a TSV file containing questions, this function will return a 
    list of `Question` objects. Depending on the `use_sample` flag, it may also try to include 
    the sample answer associated with each question

    Args:
        file (str): Path to the TSV file containing the questions
        use_sample (bool): Flag indicating whether to include the sample answer. If set to 
                           True and a sample answer is present in the TSV file, the sample 
                           answer will be added to the corresponding `Question` object

    Returns:
        List[Question]: List of parsed `Question` objects

    Notes:
        - The TSV file should contain a header which this function will skip
        - The expected format for the columns in the file is:
            Question | ScoreOffset | SampleAnswer (Optional) | QuestionId
    """
    questions = []
    with open(file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            if len(row) == 0:
                log.warning(f"Found empty row in file: {file}")
                continue
            log.debug(f'Parse Question: {row[0]}')
            parsed_row = Question(row[0])

            log.debug(f"Parse ScoreOffset: {row[1]}")
            parsed_row.score_offset = int(row[1])

            if use_sample and len(row[2]) > 1:
                log.debug(f'Parse SampleAnswer: {row[2]}')
                parsed_row.sample_answer = row[2]

            log.debug(f'Parse QuestionId: {row[-1]}')
            parsed_row.question_id = row[-1]
            questions.append(parsed_row)
    return questions


def get_answers(file:str) -> List[Answer]:
    """ Load and parse answers from a specified TSV file

    This function reads a TSV file and extracts answers, returning them as a list 
    of `Answer` objects. The function also handles answers that come with scores

    Args:
        file (str): Path to the TSV file containing the answers

    Returns:
        List[Answer]: List of parsed `Answer` objects

    Notes:
        - The TSV file should contain a header, which this function will skip
        - The expected format for the columns in the file is: 
            QuestionId | AnswerId | Answer | Score1 | Score2
    """
    answers = []
    with open(file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header row
        for row in reader:
            if len(row) == 0:
                log.warning(f"Found empty row in file: {file}")
                continue
            log.debug(f'Parse Answer: {row[2]}')
            answer = Answer(row[0], row[1], row[2])

            # Add score if applicable
            if len(row) == 5:
                answer.score_1 = row[3]
                answer.score_2 = row[4]

            answers.append(answer)

    return answers


def get_answers_per_question(file:str) -> Dict[str, List[Answer]]:
    """ Load and group answers by their question Ids from a specified TSV file

    Given a TSV file containing answers, this function categorizes each answer
    under its associated question Id, returning a dictionary where each key is
    a question Id and its associated value is a list of answers

    Args:
        file (str): Path to the TSV file containing the answers

    Returns:
        Dict[str, List[Answer]]: A dictionary where keys are question Ids and values are 
                                 lists of associated answers

    Notes:
        - The TSV file should contain a header, which this function will skip
        - The expected format for the columns in the file is: 
            QuestionId | AnswerId | Answer | Score1 | Score2
    """
    answers = {}
    answer_list = get_answers(file)

    for answer in answer_list:
        # Sort answers per question_id
        if answer.question_id in answers:
            answers[answer.question_id].append(answer)
        else:
            answers[answer.question_id] = [answer]
    return answers


# Save tsv files


def write_rated_answers_tsv(file:str, answers:List[Answer], extend:bool) -> None:
    """ Write a list of answers to the specified TSV file

    This function writes the provided list of answers to a TSV file, either appending 
    to an existing file or creating a new one based on the `extend` parameter

    Args:
        file (str): Path to the TSV file where answers will be written
        answers (List[Answer]): List of answers to write
        extend (bool): If True, the function appends to the existing file. Otherwise, it 
                       creates a new file or overwrites the existing one
    
    Notes:
        - Score1 is the rating generated by the trained model, while Score2 is the reference 
          rating from ChatGPT or Experts
        - If the TSV file does not exist or `extend` is set to False, a new file will be 
          created with headers naming all columns
        - The format for the columns in the file is:
            QuestionId | AnswerId | Score1 | Score2
    """
    mode = 'a' if extend and os.path.exists(file) else 'w'

    log.debug(f"Writing rated answers: {answers} to file: {file}")
    with open(file, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if mode == 'w':
            writer.writerow(['QuestionId', 'AnswerId', 'Score1', 'Score2'])
        for answer in answers:
            writer.writerow([answer.question_id, answer.answer_id, answer.score_1, answer.score_2])


def write_answers_tsv(file:str, answers:Iterator[List[Answer]], extend:bool) -> None:
    """ Write a list of lists of answers to the specified TSV file

    This function writes the provided list of answers to a TSV file, either appending 
    to an existing file or creating a new one based on the `extend` parameter

    Args:
        file (str): Path to the TSV file where answers will be written
        answers (answers:Iterator[List[Answer]]): Iterator of Lists of answers to write
        extend (bool): If True, the function appends to the existing file. Otherwise, it 
                       creates a new file or overwrites the existing one
    
    Notes:
        - If the TSV file does not exist or `extend` is set to False, a new file will be 
          created with headers naming all columns
        - The format for the columns in the file is:
            QuestionId | Answer | AnswerId | Score1 | Score2
    """
    mode = 'a' if extend and os.path.exists(file) else 'w'

    log.debug(f"Writing answers: {answers} to file: {file}")
    with open(file, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if mode == 'w':
            writer.writerow(['QuestionId', 'Answer', 'AnswerId', 'Score1', 'Score2'])
        for answer_list in answers:
            for answer in answer_list:
                # every answer represents one row
                writer.writerow([answer.question_id, answer.answer, answer.answer_id, answer.score_1, answer.score_2])
            csvfile.flush()
