from tsv_utils import get_answers_per_question, Answer
import logging as log
from typing import List, Tuple, Dict, Union
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from config_logger import config_logger
from config import Configuration
import os
import csv
import shutil
from io import TextIOWrapper


def _cleanup(config:Configuration) -> None:
    """ Ensures that none of the files, created by this module, already exist

    Args:
        config (Configuration): Allows access to the projects central configuration
    """
    base_folder = config.get_chat_gpt_cm_path()
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)

    os.makedirs(base_folder, exist_ok=True)


def _write_matrix(file:TextIOWrapper, matrix:List[List[int]]) -> int:
    """ Write the confusion matrix to a text file in a human-readable format

    This function takes a confusion matrix and writes it to a specified text file, formatted for easy visual inspection
    ans usage with LaTeX. Alongside the matrix, additional information, such as True Positives (TP), 
    False Negatives (FN), and total predictions per class, is included to provide more context

    Args:
        file (TextIOWrapper): The file where the matrix will be written
        matrix (List[List[int]]): The confusion matrix to be written

    Returns:
        int: The total number of predictions represented in the matrix
    """
    file.write("-------------------------------------\n")
    file.write(" | 0 | 1 | 2 | 3\n")
    file.write("-------------------\n")
    matrix_total = 0
    for i, row in enumerate(matrix):
        numbers = []
        row_total = 0
        for number in row:
            row_total += int(number)
            numbers.append(str(number).zfill(3))

        matrix_total += row_total
        if row_total > 0:
            info = f" | TP: {row[i]}, {round(100 * row[i] / row_total, 2)}%; FN: {row_total - row[i]}; Total: {row_total}\n"
        else:
            info = "\n"
        file.write(f"{i}|"+"|".join(numbers) + info)
    file.write("-------------------------------------\n")

    file.write("LaTeX:\n")
    file.write("\\begin{tabular}{|c|c|c|c|c|}\n")        
    file.write("    \\hline\n")
    file.write("        & 0 & 1 & 2 & 3 \\\\\n")
    file.write("    \\hline\n")

    for i, row in enumerate(matrix):
        numbers = []
        for number in row:
            numbers.append(str(number).zfill(3))

        file.write(f"    {i} & " + " & ".join(numbers) + " \\\\\n")
        file.write("    \\hline\n")

    file.write("\\end{tabular}\n")
    file.write("-------------------------------------\n")
    
    return matrix_total


def _calculate_performance(y_true:List[int], y_pred:List[int], average:str=None) -> Tuple[float, float, float, Union[float, None]]:
    """ Calculate performance metrics based on true and predicted labels

    Computes precision, recall, accuracy and F1-score for the given true and predicted labels. The function can calculate metrics
    for each label individually and return their average weighted by support when the 'average' parameter is specified. If
    'average' is None, the function only returns the weighted scores.

    Args:
        y_true (List[int]): List of true labels
        y_pred (List[int]): List of predicted labels
        average (str, optional): Method to calculate metrics. If None, only the weighted average scores are returned.
            This parameter is meant to handle label imbalance. Possible values include 'macro', 'micro', 'weighted', etc. 
            For instance, using 'weighted' alters 'macro' to account for label imbalance and can result in an F-score 
            that is not between precision and recall. Defaults to None

    Returns:
        Tuple[float, float, float, Union[float, None]]: A tuple containing:
            - Precision score
            - Recall score
            - F1-score
            - Accuracy (only if 'average' is specified, otherwise None)
    """
    if average:
        accuracy = accuracy_score(y_true, y_pred)
    else:
        accuracy = None

    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)

    return precision, recall, f1, accuracy


def _calculate_matrices_for_model(config:Configuration, model:str, expert_answers_per_question:Dict[str, Answer]):
    """_summary_

    Args:
        config (Configuration): _description_
        model (str): _description_
        expert_answers_per_question (Dict[str, Answer]): _description_
    """
    gpt_answers_per_question = get_answers_per_question(config.get_samples_path(f"{model}_rating_expert_data"))
    filename = config.get_path_for_chat_gpt_cm_file(f"{model}_confusion_matrices.txt")
    
    all_gpt_answers = []
    all_experts_answers = []
    with open(filename, mode='a') as file:
        for question_id, answers in gpt_answers_per_question.items():
            experts_answers = expert_answers_per_question[question_id]
            all_gpt_answers.extend(answers)
            all_experts_answers.extend(experts_answers)

            for gpt_score in [1,2]:
                for experts_score in [1,2]:
                    file.write(f"Model: {model}, Essay Set: {question_id}, score {gpt_score} vs experts score {experts_score}\n")
                    _annotations_to_cm(model, question_id, file, answers, experts_answers, gpt_score, experts_score)

        for gpt_score in [1,2]:
                for experts_score in [1,2]:
                    file.write(f"Model: {model}, score {gpt_score} vs experts score {experts_score}\n")
                    _annotations_to_cm(model, question_id, file, all_gpt_answers, all_experts_answers, gpt_score, experts_score)


def _annotations_to_cm(model:str, essay_set:str, file:TextIOWrapper, gpt_answers:List[Answer], expert_answers:List[Answer], gpt_score:int, expert_score:int) -> None:
    """ Calculate the confusion matrix and performance benchmarks for annotations from ChatGPT and human experts

    This function generates a confusion matrix and calculates the performance benchmarks
    comparing the annotations provided by ChatGPT with corresponding annotations from
    human experts. It iterates through lists of answers, compares them, and uses the
    scores to build the confusion matrix

    Args:
        gpt_answers (List[Answer]):  Predictions from ChatGPT
        expert_answers (List[Answer]): Ground truth provided by human experts
        gpt_score (int): An integer representing the desired score type to retrieve from the predictions (either 1 or 2)
        expert_score (int): An integer representing the desired score type to retrieve from the true values(either 1 or 2)

    Returns:
        Any: The confusion matrix in the form of a list of lists and the performance benchmarks
    """
    expert_ratings = []
    gpt_ratings = []
    
    for gpt_answer in gpt_answers:
        for expert_answer in expert_answers:
            if expert_answer.answer_id == gpt_answer.answer_id:
                expert_ratings.append(int(getattr(expert_answer, f'score_{expert_score}')))
                gpt_ratings.append(int(getattr(gpt_answer, f'score_{gpt_score}')))

    _write_matrix(file, confusion_matrix(expert_ratings, gpt_ratings).tolist())
    
    precisions, recalls, f1s, _ = _calculate_performance(expert_ratings, gpt_ratings)
    for category, (precision, recall, f1) in enumerate(zip(precisions, recalls, f1s)):
        _write_to_tsv(config=config, model=model, gpt_score=gpt_score, expert_score=expert_score, essay_set=essay_set, category=category, precision=precision, recall=recall, f1=f1)
        file.write(f"Category: {category}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1: {f1}\n")
        file.write("-------------------------------------\n")

    precision, recall, f1, accuracy =_calculate_performance(expert_ratings, gpt_ratings, 'weighted')
    _write_to_tsv(config, model, gpt_score, expert_score, essay_set, "All", precision, recall, f1, accuracy)
    file.write(f"Weighted across categories\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Recall: {recall}\n")
    file.write(f"F1: {f1}\n")
    file.write("###################################################\n\n")


def _write_to_tsv(config:Configuration, model:str, gpt_score:int, expert_score:int, essay_set:str, category:str, precision:float, recall:float, f1:float, accuracy:float=None) -> None:
    """ Write the given performance metrics to a tab-separated values (TSV) file

    This function takes in the metrics, model details, and other associated parameters and writes them into a TSV file.
    If the TSV file doesn't exist, a new one will be created with headers. Otherwise, the data will be appended

    Args:
        config (Configuration): Allows access to the projects central configuration
        model (str):  Model (davinci for text-davinci-003, turbo for gpt-3.5-turbo or gpt4 for GPT-4)
        gpt_score (int):  Score type used for answers annotated by ChatGPT (either 1 or 2)
        expert_score (int): Score type used for answers created by human experts (either 1 or 2)
        essay_set (str): Associated question or essay set Id
        category (str): Category of the test data or evaluation, indicating the type or domain of the test data
        precision (float): Precision metric derived from the confusion matrix
        recall (float): Recall metric derived from the confusion matrix
        f1 (float): F1 score metric derived from the confusion matrix
        accuracy (float, optional): Accuracy metric derived from the confusion matrix. If not provided,
            "N/A" will be written to the TSV file. Defaults to None

        Note:
            The TSV file will contain the following headers:
            ['Model', 'Essay Set', 'GPT Score', 'Expert Score', 'Category', 'Precision', 'Recall', 'F1', 'Accuracy']
    """
    path = config.get_path_for_chat_gpt_cm_file("performance.tsv")

    mode = 'a' if os.path.exists(path) else 'w'
    with open(path, mode, newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        if mode == 'w':
            # Header row
            writer.writerow(['Model', 'Essay Set', 'GPT Score', 'Expert Score', 'Category', 'Precision', 'Recall', 'F1', 'Accuracy'])

        writer.writerow([model, essay_set, gpt_score, expert_score, category, precision, recall, f1, accuracy if accuracy else "N/A"])


if __name__ == "__main__":
    config_logger(log.DEBUG, "results_to_confusion_matrix.log")
    config = Configuration()
    
    _cleanup(config)
    
    expert_answers_per_question = get_answers_per_question(config.get_samples_path("experts"))

    _calculate_matrices_for_model(config, "davinci", expert_answers_per_question)
    _calculate_matrices_for_model(config, "turbo", expert_answers_per_question)
    _calculate_matrices_for_model(config, "gpt4", expert_answers_per_question)