import json
from typing import List, Tuple, Union, Dict
from config import Configuration
import os
import shutil
import re
import csv
import logging as log
from config_logger import config_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from collections import defaultdict
from io import TextIOWrapper


def _recursive_default_dict() -> defaultdict:
    """ Create and return a defaultdict that produces further defaultdicts on demand

    This function returns a defaultdict. When a key is accessed that does not exist, 
    it provides another defaultdict with the same behavior, thereby allowing multi-level 
    nested access without raising KeyError

    Returns:
        defaultdict: A recursive defaultdict
    """
    return defaultdict(_recursive_default_dict)


def _matrix_to_labels(matrix) -> Tuple[List[int], List[int]]:
    """ Convert a confusion matrix into lists of true labels and predicted labels

    This function decomposes a confusion matrix into the original true labels and their corresponding predicted labels. 
    Each cell in the confusion matrix represents the count of instances where the true label is the row index and the 
    predicted label is the column index

    Args:
        matrix (List[List[int]]): The confusion matrix to be converted. Each cell (i, j) indicates the number of times
            the true label "i" was predicted as label "j"

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists:
            - The first list represents the true labels
            - The second list represents the corresponding predicted labels
    """
    y_true = []
    y_pred = []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            y_true.extend([i] * count)
            y_pred.extend([j] * count)

    return y_true, y_pred


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


def _cleanup(config:Configuration) -> None:
    """ Ensures that none of the files, created by this module, already exist

    Args:
        config (Configuration): Allows access to the projects central configuration
    """
    results_root_path = config.get_results_root_path()
    for data_file in os.listdir(results_root_path):
        if data_file.endswith("_confusion_matrices.txt"):
            full_path = os.path.join(results_root_path, data_file)
            os.remove(full_path)
    
    performance_base_folder = config.get_performance_root_path()
    if os.path.exists(performance_base_folder):
        shutil.rmtree(performance_base_folder)

    os.makedirs(performance_base_folder, exist_ok=True)


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


def _write_performance_values(precision:float, recall:float, f1:float, accuracy:float=None) -> None:
    """  Write the provided performance metrics to a text file

    This function takes precision, recall, F1 score, and optionally, accuracy, and writes them to a text file. 
    If accuracy is not provided, it will not be written to the file

    Args:
        precision (float): Precision of the predictions
        recall (float): Recall of the predictions
        f1 (float): F1 score of the predictions
        accuracy (float, optional): Overall accuracy of the predictions. If not provided, it will not be written to the file
    """
    # Accuracy is not calculated per category
    weighted = ""
    if accuracy:
        file.write(f"Accuracy: {accuracy}\n")
        weighted = " (Weighted)"

    file.write(f"Precision{weighted}: {precision}\n")
    file.write(f"Recall{weighted}: {recall}\n")
    file.write(f"F1 Score{weighted}: {f1}\n")


def _write_to_tsv(config:Configuration, metrics:List[Dict[str, Union[str, float]]], model_identifier:List[str], test_data_source:str, category:str) -> None:
    """ Write the given performance metrics to a tab-separated values (TSV) file

    This function takes in the metrics, model details, and other associated parameters and writes them into a TSV file.
    If the TSV file doesn't exist, a new one will be created with headers. Otherwise, the data will be appended

    Args:
        config (Configuration): Allows access to the projects central configuration
        metrics (List[Dict[str, Union[str, float]]]): A list of dictionaries where each dictionary contains the label and value for a performance metric
        model_identifier (List[str]): A list of identifiers providing details about the model. Expected order is:
                                      - Model type ('bert' for BERT, 'xgb' for XG-Boost)
                                      - Associated question or essay set Id
                                      - Variation Id of the model
                                      - Number of samples used for training
                                      - Source of the training samples
        test_data_source (str): Source or type of the test data used for generating the metrics
        category (str): Category of the test data or evaluation, indicating the type or domain of the test data

    """
    path = config.get_path_for_performance_file("performance.tsv")

    mode = 'a' if os.path.exists(path) else 'w'
    with open(path, mode, newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        if mode == 'w':
            # Header row
            writer.writerow(['Model Type', 'Question Id', 'Batch Size', 'Variation Id', 'Training Data Source', 'Category', 'Tested With', 'Category'] + [metric['label'] for metric in metrics])

        #xgb, 5, 3200, A, gpt4, experts-testing, 0, metric_a ... metric_n
        writer.writerow(model_identifier + [test_data_source, category] + [metric['value'] for metric in metrics])


def _to_metric_list(precision:float, recall:float, f1:float, accuracy:Union[float, None]) -> List[Dict[str, str]]:
    """ Convert the provided performance metrics to a list of dictionaries

    Each metric is represented as a dictionary with two keys: "label" (the name of the metric) and "value" (the corresponding value). 
    If the accuracy is None, its value will be represented as "N/A"

    Args:
        precision (float): Precision value of the predictions
        recall (float): Recall value of the predictions
        f1 (float): F1 score value of the predictions
        accuracy (Union[float, None]): Accuracy value of the predictions. If not provided, it is represented as "N/A"

    Returns:
        List[Dict[str, Union[str, float]]]: A list of dictionaries representing the metrics, with each dictionary containing a label and a value
    """
    return [
        {
            "label": "Precision",
            "value": precision
        },
        {
            "label": "Recall",
            "value": recall
        },
        {
            "label": "F1",
            "value": f1
        },
        {
            "label": "Accuracy",
            "value": accuracy if accuracy else "N/A"
        }
    ]


def _add_entry(dictionary:Dict[str, List[float]], name:str, entry:float) -> None:
    """ Append an entry to a list within a dictionary under a specified key
    
    If the key exists in the dictionary, the entry is appended to the list under that key.
    If the key does not exist, a new list with the entry as the first element is created under that key
    
    Args:
        dictionary (Dict[str, List[float]]): The dictionary to which the entry will be added
        name (str): The key in the dictionary where the entry will be added. If the key doesn't exist, 
                    it will be created
        entry (float): The value to be appended to the list corresponding to the provided key in the dictionary
    """
    if name in dictionary:
        dictionary[name].append(entry)
    else:
        dictionary[name] = [entry]


def _print_boxplot(model_type:str, training_data_source:str, test_data_source:str, question_id:str, metric:str, x_y_boxplot_data:Dict[str, List[float]], category:Union[str, None]) -> None:
    """ Generate and save a Boxplot visualizing the distribution of a specific performance metric

    The Boxplot displays different batch sizes, based on the training and testing data sources, the model type,
    the associated essay set (question Id), and an optional category

    Args:
        model_type (str): Type of the model. Examples include 'bert' for BERT and 'xgb' for XG-Boost
        training_data_source (str): The source of the Samples, the model was trained with
        test_data_source (str): The source of the Samples, the model was tested with
        question_id (str): Id of the associated essay set
        metric (str): The performance metric being visualized (e.g., 'precision', 'recall')
        x_y_boxplot_data (Dict[str, List[float]]): Mapping of labels (typically batch sizes) to lists of values for the metric
        category (Union[str, None]): Optional category associated with the metric, if applicable. If not provided, the metric is assumed to be weighted

    Note:
        The Boxplot is saved as a PDF file. The filename format is determined by the combination of training and testing sources, question Id, model type, metric, and category
    """
    figure, ax = plt.subplots()

    labels = list(x_y_boxplot_data.keys())
    box_data = [x_y_boxplot_data[key] for key in labels]

    ax.boxplot(box_data, vert=True, patch_artist=True)
    ax.set_xticklabels(labels)

    test_data = test_data_source.split("-")

    category_title = f" of category {category}" if category else " (Weighted)"
    ax.set_title(f"""{metric}{category_title} of {model_type.upper()} models trained with Data generated by {training_data_source}
tested with data generated by {test_data[0]} for Essay Set {question_id} used for {test_data[0]}.""")
    
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(metric)

    # One Diagram per training_data_source, test_data_source, question_id, model_type and metric
    filename = f"boxplot_performance_{training_data_source}_{test_data_source}_{question_id}_{model_type.upper()}_{metric}_{category}_.pdf"
    figure.savefig(config.get_path_for_performance_file(filename), bbox_inches='tight')


if __name__ == "__main__":
    config = Configuration()
    config_logger(log.DEBUG, "analyse_confusion_matrices.log")

    results_root_path = config.get_results_root_path()
    _cleanup(config)

    for data_file in os.listdir(results_root_path):

        if not data_file.endswith("_confusion_matrices.json"):
            continue

        match = re.search(r'_([a-zA-Z]+-[a-zA-Z]+)_', data_file)
        if match:
            test_data_source = match.group(1)
        else:
            log.error(f"Unable to parse test_data_source from: {data_file}")
            continue

        allocated_metrics = _recursive_default_dict()
        text_file_name = data_file.replace(".json", ".txt")

        with open(config.get_path_for_results_file(data_file), "r") as file:
            data = json.load(file)
        
        # This values are equal for all keys in the processed file
        training_data_source = ""
        question_id = ""

        with open(config.get_path_for_results_file(text_file_name), 'w') as file:
            for key, entry in data.items():
                file.write(f"Key: {key}\n")
                file.write(f"Path: {entry['path']}\n")

                # example key: xgb_5_3200_A_gpt4
                # model_type, question_id, batch_size, batch_id, training_data_source
                model_identifier = key.split("_")
                question_id = model_identifier[1]
                batch_size = model_identifier[2]
                training_data_source = model_identifier[4]


                matrix_total = _write_matrix(file, entry["cm_matrix"])
                label = _matrix_to_labels(entry["cm_matrix"])
                
                precisions, recalls, f1s, _ = _calculate_performance(*label)
                for category, (precision, recall, f1) in enumerate(zip(precisions, recalls, f1s)):
                    _add_entry(allocated_metrics[model_identifier[0]][category]["precision"], batch_size, precision)
                    _add_entry(allocated_metrics[model_identifier[0]][category]["recall"], batch_size, recall)
                    _add_entry(allocated_metrics[model_identifier[0]][category]["f1"], batch_size, f1)

                    file.write(f"Performance for category: {category}\n")
                    _write_to_tsv(config, _to_metric_list(precision, recall, f1, None), model_identifier, test_data_source, category)

                    _write_performance_values(precision, recall, f1)
                    file.write("------------------------------------------\n")

                file.write(f"Number of Samples: {matrix_total}\n")

                precision, recall, f1, accuracy =_calculate_performance(*label, 'weighted')
                _add_entry(allocated_metrics[model_identifier[0]][None]["accuracy"], batch_size, accuracy)
                _add_entry(allocated_metrics[model_identifier[0]][None]["precision"], batch_size, precision)
                _add_entry(allocated_metrics[model_identifier[0]][None]["recall"], batch_size, recall)
                _add_entry(allocated_metrics[model_identifier[0]][None]["f1"], batch_size, f1)
                
                _write_performance_values(precision, recall, f1, accuracy)
                _write_to_tsv(config, _to_metric_list(precision, recall, f1, accuracy), model_identifier, test_data_source, "weighted")
                file.write("###################################################\n\n")
            
            for model_type, sub_dict_a in allocated_metrics.items():
                for category, sub_dict_b in sub_dict_a.items():
                    for metric, sub_dict_c in sub_dict_b.items():
                        x_y_boxplot_data = {}
                        for batch_size, values in sub_dict_c.items():
                            x_y_boxplot_data[batch_size] = values

                        _print_boxplot(model_type, training_data_source, test_data_source, question_id, metric, x_y_boxplot_data, category)