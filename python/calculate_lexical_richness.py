import logging as log
from typing import List, Dict, Tuple, Any, Callable, Union

import numpy as np
from numpy.typing import NDArray
import os
from tsv_utils import Answer, get_answers_per_question
from lexicalrichness import LexicalRichness

from config_logger import config_logger
from config import Configuration
import csv
import random
import shutil
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Source: https://lexicalrichness.readthedocs.io/en/latest/#lexicalrichness.LexicalRichness.ttr
class LRCalculator:
    """ Helper class for calculating the lexical richness of answers

    This class uses various algorithms to measure the lexical richness or diversity of the given answers. It also 
    provides safeguards against problematic calculations or conditions that might cause errors or inaccuracies.

    Attributes:
        lex (LexicalRichness): An instance of the LexicalRichness class initialized with the input words
        methods (List[str]): A list of methods or algorithms to compute lexical diversity
        not_one (List[str]):  Methods that may divide by zero if the number of words is exactly one
        not_equal (List[str]): Methods that may divide by zero if the number of words equals the number of terms
        special (Dict[str, List[Dict[str, int]]]): Specific conditions where certain algorithms don't function correctly
    """
    # The order of _LABELS must match the order of _METHODS
    _LABELS = ["Number of Terms", "Number of Words", "Type-Token Ratio", "Root Type-Token Ratio (Guiraud's R)",
              "Corrected Type-Token Ratio", "Herdan’s C", "Summer", "Maas", "Dugast", "Yule's K", "Yule's I", "Herdan's VM", "Simpson's D"]

    _METHODS = ['terms', 'words', 'ttr', 'rttr', 'cttr', 'Herdan', 'Summer', 'Maas', 'Dugast', 'yulek', 'yulei', 'herdanvm', 'simpsond']


    @classmethod
    def LABELS(self) -> List[str]:
        return self._LABELS


    def __init__(self, words:str):
        """Initializes the LRCalculator with the given words."""
        self.lex = LexicalRichness(words)
        
        # divide for example by the logarithm of the word count
        self.not_one = ['Herdan', 'Summer', 'Maas', 'simpsond']

        # divide by zero if words equals terms
        self.not_equal = ['Dugast', 'yulei']

        # particular conditions that do not work for certain algorithms
        self.special = {
            "herdanvm": [{
                "words": 19,
                "terms": 19
            }, {
                "words": 21,
                "terms": 21
            }, {
                "words": 38,
                "terms": 38
            }]
        }


    def _safe_calculation(self, calculation:Callable[[], float], method:str) -> Union[float, Any]:
        """ Executes the given calculation while handling potential problematic scenarios

        Args:
            calculation (Callable[[], float]): The actual function or calculation to be executed
            method (str): The name of the method being executed to determine which scenarios to check for

        Returns:
            Union[float, Any]: Result of the calculation or np.nan if the value is problematic
        """
        log.info(f"Calculate method: {method}. Words: {self.lex.words}, Terms: {self.lex.terms}")

        # These methods divide by log(words)
        if method in self.not_one and self.lex.words == 1:
            log.info(f"Skip calculation - it can't be performed with only one word: {method}. Words: {self.lex.words}, Terms: {self.lex.terms}")
            return np.nan

        # These methods divide by words - terms
        if method in self.not_equal and self.lex.words == self.lex.terms:
            log.info(f"Skip calculation - it can't be performed if words and terms are equal in number: {method}. Words: {self.lex.words}, Terms: {self.lex.terms}")
            return np.nan

        # certain conditions that do not work
        if method in self.special:
            for condition in self.special[method]:
                if self.lex.words == condition["words"] and self.lex.terms == condition["terms"]:
                    log.info(f"Skip calculation - it can't be performed because of a special condition: {method}. Words: {self.lex.words}, Terms: {self.lex.terms}")
                    return np.nan

        try:
            return calculation()
        except Exception as e:
            log.error(f"Exception in _safe_calculation: {str(e)}, for method: {method}. Words: {self.lex.words}, Terms: {self.lex.terms}")
            return np.nan


    @property
    def data(self) -> Tuple[float, ...]:
        """ Computes lexical diversity for the text using all defined algorithms

        Returns:
            Tuple[float, ...]: Tuple of lexical diversity measurements from all methods
        """
        return tuple(self._safe_calculation(lambda: getattr(self.lex, method), method) for method in self._METHODS)

    @staticmethod
    def _find_min_and_max_values(metrics:List[Tuple[float, ...]]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """ Compute the minimum and maximum lexical richness values for each metric across all answers

        For each metric in the list of tuples, this method calculates the minimal and maximal
        lexical richness, ignoring any NaN values

        Args:
            metrics (List[Tuple[float, ...]]): A list of tuples, where each tuple represents the 
                lexical richness metrics of an answer

        Returns:
            Tuple[Tuple[float, ...], Tuple[float, ...]]: A tuple of two tuples. The first tuple
                contains the minimum lexical richness values for each metric, and the second tuple
                contains the maximum lexical richness values for each metric
        """
        min_lr = tuple(
            min((value for value in group if not np.isnan(value))) for group in zip(*metrics)
        )

        max_lr = tuple(
            max((value for value in group if not np.isnan(value))) for group in zip(*metrics)
        )
        return min_lr, max_lr

    @staticmethod
    def calculate_lr(answers:List[Answer]) -> Tuple[float, Any, Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
        """ Computes lexical richness for a list of answers both individually and as a whole

        Args:
            answers (List[Answer]): List of answers for which lexical diversity should be computed

        Returns:
            Tuple[float, Any, Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]: A tuple containing:
            - Average lexical diversity per answer
            - Count of answers not used in calculations due to specific conditions
            - Lexical diversity of the combined answers
            - The minimal lexical diversity for each metric across all answers
            - The maximal lexical diversity for each metric across all answers
        """
        average_lr = [LRCalculator(answer.answer).data for answer in answers]
        average_lr_mask = np.isnan(average_lr)

        min_lr, max_lr = LRCalculator._find_min_and_max_values(average_lr)

        return np.nanmean(np.array(average_lr), axis=0), np.sum(average_lr_mask, axis=0), LRCalculator(' '.join([answer.answer for answer in answers])).data, min_lr, max_lr


class LRFigures:
    """ A utility class for plotting diagrams related to lexical richness metrics

    Attributes:
        _COLORS (List[str]): A list of color codes for the diagrams
        _COLOR_NAMES (List[str]): A list of color names corresponding to `_COLORS`
    """

    _COLORS = ['r', 'b', 'g', 'c', 'm']
    _COLOR_NAMES = ['red', 'blue', 'green', 'cyan', 'magenta']


    def __init__(self, data_source:str) -> None:
        """ Initializes the LRFigures with a given data source

        Args:
            data_source (str): Source of the data
        """
        self.data_source = data_source
        self.metrics = {}


    def plot(self, metrics:List[str], number_of_answers:str, values:List[float], question_id:str) -> None:
        """ Updates the internal metrics dictionary with given metrics and values for a specific question
        
        Every metric has its own diagram. The values for all questions accumulate in a single diagram

        Args:
            metrics (List[str]): Labels for the metrics, used to identify the targeted diagram
            number_of_answers (str): The number of answers, corresponding to the x-axis label in the diagram
            values (List[float]): x Data values for all metrics
            question_id (str): Identifier for the associated question
        """

        for index, metric in enumerate(metrics):
            if not metric in self.metrics:
                self.metrics[metric] = {}

            if question_id in self.metrics[metric]:
                self.metrics[metric][question_id]['labels'].append(number_of_answers)
                self.metrics[metric][question_id]['values'].append(values[index])
            else:
                self.metrics[metric][question_id] = {
                    "labels": [number_of_answers],
                    "values": [values[index]]
                }


    @staticmethod
    def _filter(x:NDArray[Any], y:NDArray[Any], y_pred:NDArray[Any], tolerance:int=1) -> Tuple[NDArray[Any], NDArray[Any]]:
        """ Filters data points based on their distance from the predicted graph

        This static method filters out data points that are too distant from the
        predicted graph within a specified tolerance level. The distance is calculated
        based on the difference between actual y values and predicted y values. Points
        that fall within the acceptable distance (defined by the tolerance level) from
        the predicted graph are retained

        Args:
            x (NDArray[Any]): An array representing the x-coordinates of the data points
            y (NDArray[Any]): An array representing the actual y-coordinates of the data points
            y_pred (NDArray[Any]): An array representing the predicted y-coordinates of the graph
            tolerance (int, optional): The maximum acceptable distance from the graph,
                expressed as a percentage. Defaults to 1

        Returns:
            Tuple[NDArray[Any], NDArray[Any]]:  A tuple containing arrays of filtered x and y
                coordinates, determined based on the provided predictions and tolerance level
        """
        y_diff_pred = np.diff(y_pred)
        y_diff = np.diff(y - y_pred)
        y_diff = np.append(y_diff, y_diff[-1])

        thresholds = tolerance * np.abs(y_diff_pred)
        thresholds = np.append(thresholds, thresholds[-1])
        mask = np.abs(y_diff) <= thresholds

        return x[mask], y[mask]
    

    @staticmethod
    def _get_coordinates_for_graph(x:NDArray[Any], y:NDArray[Any], graph_callback:Callable) -> Tuple[NDArray[Any], NDArray[Any]]:
        """ Attempts to approximate the provided x and y values to a specified graph
        using a provided callback function for graph calculations

        This method aims to fit the provided x and y data points to a graph generated
        using the `graph_callback`. It initially performs a curve fit for the original
        data and applies a filtering method to refine the points. If sufficient points
        remain post-filtering, a smoothed, fitted graph is generated and returned

        Args:
            x (NDArray[Any]): An array of x-coordinates of the data points
            y (NDArray[Any]): An array of y-coordinates of the data points
            graph_callback (Callable): A callback function used for calculating 
                the values of the approximated graph

        Returns:
            Tuple[NDArray[Any], NDArray[Any]]: A tuple containing arrays of x and y 
                coordinates for the approximated graph, or None, None if the approximation 
                could not be calculated
        
        Note:
            The `graph_callback` should be compatible with the curve fitting method and be able
            to accept the x values and any fitting parameters
        """
        try:
            popt, _ = curve_fit(graph_callback, x, y)
            y_pred = graph_callback(x, *popt)

            x_filtered, y_filtered = LRFigures._filter(x, y, y_pred)
            
            if len(x_filtered) < 0.75 * len(x):
                log.error(f"Not enough values left to calculate a graph: {len(x_filtered)} of {len(x)}")
                return None, None
            
            x_smooth_filtered = np.linspace(min(x_filtered), max(x_filtered), 100)
            popt_filtered, _ = curve_fit(graph_callback, x_filtered, y_filtered)
            return x_smooth_filtered, graph_callback(x_smooth_filtered, *popt_filtered)
        except RuntimeError:
            return None, None
    

    @staticmethod
    def _logarithm(x:NDArray[Any], y:NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        """ Attempts to approximate a logarithmic graph using given x and y coordinates.

        This method leverages a predefined lambda function (a * np.log(b * x)) with 
        curve fitting techniques to approximate a logarithmic graph. The fitted graph 
        is derived by employing the _get_coordinates_for_graph method, which takes the provided x and 
        y coordinates along with the logarithmic function as inputs, subsequently returning 
        the coordinates of the approximated graph

        Args:
            x (NDArray[Any]): An array of x-coordinates of the data points
            y (NDArray[Any]): An array of y-coordinates of the data points

        Returns:
            Tuple[NDArray[Any], NDArray[Any]]: A tuple containing arrays of x and y 
                coordinates for the approximated graph, or None, None if the approximation 
                could not be calculated
        
        Note:
            The method may return (None, None) indicating the inability to approximate
            the graph based on provided coordinates and the logarithmic function
        """
        return LRFigures._get_coordinates_for_graph(x, y, lambda x, a, b: a * np.log(b * x))
    

    @staticmethod
    def _inverse_logarithm(x:NDArray[Any], y:NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        """ Attempts to approximate a logarithmic graph using given x and y coordinates.

        This method leverages a predefined lambda function (a + b / np.log(x)) with 
        curve fitting techniques to approximate a inverse logarithmic graph. The fitted graph 
        is derived by employing the _get_coordinates_for_graph method, which takes the provided x and 
        y coordinates along with the inverse logarithmic function as inputs, subsequently returning 
        the coordinates of the approximated graph

        Args:
            x (NDArray[Any]): An array of x-coordinates of the data points
            y (NDArray[Any]): An array of y-coordinates of the data points

        Returns:
            Tuple[NDArray[Any], NDArray[Any]]: A tuple containing arrays of x and y 
                coordinates for the approximated graph, or None, None if the approximation 
                could not be calculated
        
        Note:
            The method may return (None, None) indicating the inability to approximate
            the graph based on provided coordinates and the logarithmic function
        """
        return LRFigures._get_coordinates_for_graph(x, y, lambda x, a, b: a + b/np.log(x))


    @staticmethod
    def _polymorph(x:NDArray[Any], y:NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        """ Approximate a polynomial graph using the given x and y coordinates.

        This method utilizes numpy's polyfit to estimate a polynomial graph based on the
        provided x and y coordinates. Initially, the method filters the data points to 
        ones close to the initial polynomial fit, using an adjustable tolerance. It seeks 
        to maintain at least 75% of the original data points by incrementally increasing 
        the tolerance if needed. Finally, it returns x and y coordinates derived from the 
        approximated polynomial for a smoothed x range

        Args:
            x (NDArray[Any]): An array of x-coordinates of the data points
            y (NDArray[Any]): An array of y-coordinates of the data points

        Returns:
            Tuple[NDArray[Any], NDArray[Any]]: A tuple containing arrays of x and y 
                coordinates for the approximated graph
        
        Notes:
        - The method employs a second-degree polynomial for the initial approximation
        - Consider that substantial increments in tolerance might affect the reliability 
          of the approximation
        """
        coefficients = np.polyfit(x, y, 2)
        polynomial = np.poly1d(coefficients)
        y_pred = polynomial(x)

        tolerance = 0.5
        x_filtered, y_filtered = LRFigures._filter(x, y, y_pred, tolerance)
        while len(x_filtered) < 0.75 * len(x):
            tolerance += 0.1
            x_filtered, y_filtered = LRFigures._filter(x, y, y_pred, tolerance)
            log.error(f"More than 25% of all x values will be ignored: {len(x_filtered)} of {len(x)}")

        x_smooth_filtered = np.linspace(min(x_filtered), max(x_filtered), 100)
        coefficients = np.polyfit(x_filtered, y_filtered, 2)
        polynomial = np.poly1d(coefficients)
        return x_smooth_filtered, polynomial(x_smooth_filtered)


    def save(self, config:Configuration) -> None:
        """ Stores the plotted diagrams as PDF files based on the current metrics

        Args:
            config (Configuration): Allows access to the projects central configuration
        """
        for metric_name, sub_dict in self.metrics.items():
            title = f"Lexical Diversity of samples generated by {self.data_source}\nThe displayed metric is {metric_name}"
        
            figure = plt.figure()
            plt.title(title)

            plt.xticks(rotation=45, ha='right')
            x_label = "Number of Answers\n"
            x_label += "_"*80
            
            i = 0
            log.debug(f"Plot metric: {metric_name}")
            for question_id, metric in sub_dict.items():
                color = self._COLORS[i]
                log.debug(f"Plot color: {color} for question: {question_id} with i: {i}")

                x = np.array([int(i) for i in metric["labels"]])
                y = np.array(metric["values"])

                x_smooth, y_smooth = LRFigures._logarithm(x, y)
                if (y_smooth is None):
                    x_smooth, y_smooth = LRFigures._inverse_logarithm(x, y)

                if (y_smooth is None):
                    x_smooth, y_smooth = LRFigures._polymorph(x, y)
    
                plt.plot(x_smooth, y_smooth, f"{color}-", alpha=0.5, linewidth=0.2)
                plt.plot(metric["labels"], metric["values"], f"{color}o")

                x_label += f"\nEssay Set {question_id} is represented as {self._COLOR_NAMES[i]}"
                i += 1

            plt.xlabel(x_label)
            plt.ylabel(metric_name)
            
            filename = f"lexical_richness_{self.data_source}_{metric_name}.pdf"
            log.info(f"Save Diagram: {filename}")

            figure.savefig(config.get_path_for_lr_file(filename), bbox_inches='tight')
            plt.close()


def _calculate_and_save_lexical_richness(path, answers_per_question:Dict[str, Answer], data_source:str, number_of_answers:Union[int, None], diagrams:LRFigures) -> None:
    """ Compute metrics of lexical richness/diversity and save them to a TSV file.

    The function calculates various lexical richness metrics for a set of answers and saves 
    the results in a TSV file. If the file already exists, the results are appended; otherwise, 
    a new file is created.

    Args:
        path (str): Path to the TSV file where the results should be saved
        answers_per_question (Dict[str, Answer]): Answers categorized by their respective questions
        data_source (str): The origin or source of the provided data
        number_of_answers (Union[int, None]): The number of answers to use, if this value is None, all answers will be used. Default is None.

    Notes:
        The lexical richness metrics include "Number of Terms", "Type-Token Ratio", "Root Type-Token Ratio", 
        and several others. The function computes both total and average values for these metrics.
    """
    labels = LRCalculator.LABELS()
    mode = 'a' if os.path.exists(path) else 'w'
    with open(path, mode, newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        if mode == 'w':
            # Header row
            writer.writerow(['Question', 'Number of Answers', 'Represents'] + labels + [label + ' Skipped' for label in labels])

        for question, answers_for_question in answers_per_question.items():
            if number_of_answers:
                if len(answers_for_question) < number_of_answers:
                    return

                answers = _get_random_answers(answers_for_question, number_of_answers)
            else:
                answers = answers_for_question

            average_lr, skipped_answers, total_lr, min_lr, max_lr = LRCalculator.calculate_lr(answers)

            diagrams.plot(labels, len(answers), total_lr, question)

            # Total for all answers
            writer.writerow([question, len(answers), data_source + ' Total'] + list(total_lr) + ['N/A']*len(labels))

            # Average per answer
            writer.writerow([question, len(answers), data_source + ' Average'] + list(average_lr) + list(skipped_answers))

            # Minimum across all answer
            writer.writerow([question, len(answers), data_source + ' Minimum'] + list(min_lr) + list(skipped_answers))

            # Maximum across all answer
            writer.writerow([question, len(answers), data_source + ' Maximum'] + list(max_lr) + list(skipped_answers))


def _get_random_answers(answers:List[Answer], number:int) -> List[Answer]:
    """ Select a random Set of Answers from a given List

    Args:
        answers (List[Answer]): The List of Answers to select form
        number (int): The number of Answers to be selected

    Returns:
        List[Answer]: Randomly selected Answers
    """
    random.shuffle(answers)
    return answers[:number]


def _cleanup(config:Configuration) -> None:
    """ Ensures that none of the files, created by this module, already exist

    Args:
        config (Configuration): Allows access to the projects central configuration
    """
    base_folder = config.get_lr_root_path()
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)

    os.makedirs(base_folder, exist_ok=True)


if __name__ == "__main__":
    config = Configuration()
    config_logger(log.WARNING, "calculate_lexical_richness.log")
    lr_file_path = config.get_lr_calculations_path()

    _cleanup(config)

    davinci_answers_per_question = get_answers_per_question(config.get_samples_path("davinci"))
    gpt4_answers_per_question = get_answers_per_question(config.get_samples_path("gpt4"))
    man_answers_per_question = get_answers_per_question(config.get_samples_path("experts"))

    davinci_diagrams = LRFigures("davinci")
    gpt4_diagrams = LRFigures("gpt4")
    experts_diagrams = LRFigures("experts")
    for size in [None, 100, 200, 400, 800, 1200, 1600, 2400, 3200]:
        _calculate_and_save_lexical_richness(lr_file_path, davinci_answers_per_question, "text-davinci-003", size, davinci_diagrams)
        _calculate_and_save_lexical_richness(lr_file_path, gpt4_answers_per_question, "gpt4", size, gpt4_diagrams)
        _calculate_and_save_lexical_richness(lr_file_path, man_answers_per_question, "experts", size, experts_diagrams)
    
    davinci_diagrams.save(config)
    gpt4_diagrams.save(config)
    experts_diagrams.save(config)

