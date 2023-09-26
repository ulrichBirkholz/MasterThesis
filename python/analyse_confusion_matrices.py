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

# TODO: documentation

def _recursive_default_dict() -> defaultdict:
    """ Create and return a defaultdict that produces further defaultdicts on demand

    This function returns a defaultdict. When a key is accessed that does not exist, 
    it provides another defaultdict with the same behavior, thereby allowing multi-level 
    nested access without raising KeyError

    Returns:
        defaultdict: A recursive defaultdict
    """
    return defaultdict(_recursive_default_dict)


def _matrix_to_labels(matrix):
    y_true = []
    y_pred = []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            y_true.extend([i] * count)
            y_pred.extend([j] * count)

    return y_true, y_pred


def _calculate_performance(y_true:List[int], y_pred:List[int], average:str=None) -> Tuple[float, float, float, Union[float, None]]:
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


def _write_matrix(file:str, matrix:List[List[int]]) -> int:
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
    return matrix_total


def _write_performance_values(precision:float, recall:float, f1:float, accuracy:float=None) -> None:
    # Accurracy is not calculated per category
    if accuracy:
        file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Precision (Weighted): {precision}\n")
    file.write(f"Recall (Weighted): {recall}\n")
    file.write(f"F1 Score (Weighted): {f1}\n")


def _write_to_tsv(config:Configuration, metrics:List[Dict[str, Union[str, float]]], model_identifier:List[str], test_data_source:str, category:str) -> None:
    path = config.get_path_for_performance_file("performance.tsv")

    mode = 'a' if os.path.exists(path) else 'w'
    with open(path, mode, newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        if mode == 'w':
            # Header row
            writer.writerow(['Model Type', 'Question Id', 'Batch Size', 'Variation Id', 'Training Data Source', 'Category', 'Tested With', 'Category'] + [metric['label'] for metric in metrics])

        #xgb, 5, 3200, A, gpt4, experts-testing, 0, metric_a ... metric_n
        writer.writerow(model_identifier + [test_data_source, category] + [metric['value'] for metric in metrics])


def _to_metric_list(precision:float, recall:float, f1:float, accuracy:Union[float, None]):
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
    if name in dictionary:
        dictionary[name].append(entry)
    else:
        dictionary[name] = [entry]


def _print_boxplot(model_type:str, training_data_source:str, test_data_source:str, question_id:str, metric:str, x_y_boxplot_data:Dict[str, List[float]], category:Union[str, None]) -> None:
    figure, ax = plt.subplots()

    # sort x ascending
    #x_y_boxplot_data_sorted = dict(sorted(x_y_boxplot_data.items()))

    # sort each individual y value ascending
    #ax.boxplot([sorted(y) for y in x_y_boxplot_data_sorted.values()])
    #ax.set_xticklabels(list(x_y_boxplot_data_sorted.keys()))

    log.error(f"data: {x_y_boxplot_data}")
    labels = list(x_y_boxplot_data.keys())
    box_data = [x_y_boxplot_data[key] for key in labels]
    log.error(f"box_data: {box_data}")
    ax.boxplot(box_data, vert=True, patch_artist=True)
    ax.set_xticklabels(labels)

    test_data = test_data_source.split("-")

    category_title = f" of category {category}" if category else " (Weighted)"
    ax.set_title(f"""{metric}{category_title} of {model_type.upper()} models trained with Data generated by {training_data_source}
tested with data generated by {test_data[0]} for Essey Set {question_id} used for {test_data[0]}.""")
    
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