from config import Configuration
from tsv_utils import get_answers, Answer
from typing import List, Tuple, Dict, Any
import logging as log
from config_logger import config_logger
import csv
import re
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import defaultdict
import os
import shutil


class KappaFigure():
    """ Represents a diagram for displaying the Quadratic Weighted Kappa (QWK) of a particular variant

    This class is designed to facilitate the creation, handling, and saving of diagrams 
    displaying QWK values associated with specific datasets or model variants. 
    It also provides utilities to write these values into a TSV file

    Attributes:
        answer_count (int): The count of answers considered. Initially set to -1
        all_counts_equal (bool): True if all answer counts are equal, otherwise False
        filename (str): Name of the file to save the diagram to
        title (str): Title for the diagram
        kappa_values (List[float]): List to store the QWK values
        labels (List[str]): List to store corresponding labels for QWK values
    """
    def __init__(self, title:str, fileLabel:str) -> None:
        """ Initialize KappaFigure with a title and file label

        Args:
            title (str): Title for the diagram
            fileLabel (str): Label for generating the file name
        """
        self.answer_count = -1
        self.all_counts_equal = True

        self.filename = f"kappa_{fileLabel}.pdf"
        self.title = title

        self.kappa_values = []
        self.labels = []
    

    def get_values(self) -> Tuple[List[str], List[float]]:
        """ Fetch all label (x-values) and QWK (y-values) of the diagram

        Returns:
            Tuple[List[str], List[float]]: Tuple containing the labels and the QWK values
        """
        return self.labels, self.kappa_values


    def plot(self, label:str, kappa:float, answer_count:int) -> None:
        """ Append a new QWK value and its corresponding label to the diagram

        Args:
            label (str): Label to be added
            kappa (float): QWK value to be appended
            answer_count (int): Number of answers considered for calculating the QWK
        """
        if kappa <= -1 or kappa >= 1:
            log.warning(f"Ignore invalid kappa: {kappa}")
            return

        self.kappa_values.append(kappa)
        self.labels.append(label)
        log.debug(f"Extending diagram: {self.filename} with kappa: {kappa} for label: {label}")

        if self.answer_count == -1:
            self.answer_count = answer_count
        elif self.answer_count != answer_count:
            # should not happen the data might be inconsistent
            self.all_counts_equal = False
            log.error(f"Data is potentially inconsistent")

            # find the smallest value
            if self.answer_count > answer_count:
                self.answer_count = answer_count


    def _write_to_tsv(self, config:Configuration) -> None:
        """ Store the QWK values into a TSV file for further review

        Args:
            config (Configuration): Allows access to the projects central configuration
        """
        file_path = config.get_qwk_result_path()
        batches = config.get_batches()
        max_number_of_labels = len(batches)
        mode = 'a' if os.path.exists(file_path) else 'w'
        with open(file_path, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            if mode == 'w':
                # Header row
                writer.writerow(['Source'] + [batch.size for batch in batches])
            
            writer.writerow([self.filename] + self.kappa_values + ['N/A']*(max_number_of_labels - len(self.labels)))


    def save(self, config:Configuration, figtext:str=None, compact:bool=False) -> None:
        """ Store the diagram as a PDF file and the labels and QWK values as a TSV file

        Args:
            config (Configuration): Allows access to the projects central configuration
            figtext (str, optional): Additional descriptive text to be included in the diagram. Defaults to None
            compact (bool, optional): If True, modifies the label display to improve readability for larger labels. Defaults to False
        """
        self._write_to_tsv(config)
        figure = plt.figure()
        plt.title(self.title)

        if compact:
            plt.xticks(rotation=45, ha='right')

        x_label = "Amount of samples the model was trained with\n"
        x_label += "_"*80
        if figtext is not None:
            x_label += f"\n{figtext}"
        elif self.all_counts_equal is True:
            x_label += f"\nEach value represents the agreement of exactly {self.answer_count} ratings"
        else:
            x_label += f"\nEach value represents the agreement of at leased {self.answer_count} ratings"

        plt.xlabel(x_label)
        plt.ylabel("Kappa")

        plt.plot(self.labels, self.kappa_values, 'ro')

        log.debug(f"Save Diagram: {self.filename}")
        figure.savefig(config.get_path_for_qwk_file(self.filename), bbox_inches='tight')


def _add_figtext(fig:Figure, figtext:str) -> None:
    """ Add specified text to the bottom of the given figure

    This function appends the provided text to the lower part of the provided 
    figure. The text is center-aligned and preceded by an underscore line

    Args:
        fig (Figure): Target matplotlib figure to which the text will be added
        figtext (str): Text to be appended to the figure

    Notes:
        - The function adjusts the figure's bottom margin to ensure the text fits
        - An underscore line (_____) is added right above the provided text for visual separation
    """
    fig.subplots_adjust(bottom=0.2)

    fig.text(0.5, 0.1, '_'*80, ha='center')
    fig.text(0.5, 0.08, figtext, ha='center', va='top')


def _recursive_default_dict() -> defaultdict:
    """ Create and return a defaultdict that produces further defaultdicts on demand

    This function returns a defaultdict. When a key is accessed that does not exist, 
    it provides another defaultdict with the same behavior, thereby allowing multi-level 
    nested access without raising KeyError

    Returns:
        defaultdict: A recursive defaultdict
    """
    return defaultdict(_recursive_default_dict)


def _add_list_entry(dict:defaultdict[str, List[Any]], name:str, entry:Any) -> None:
    """ Adds an entry to a list located by the provided key within the defaultdict.
    If the list does not already exist under that key, it is created

    Args:
        dict (defaultdict[str, List[Any]]): The defaultdict where lists are stored
        name (str): The key corresponding to the list in the defaultdict
        entry (Any): The item to be appended to the list
    """
    if name in dict:
        dict[name].append(entry)
    else:
        dict[name] = [entry]


def _get_diagram_title_model_a_score_1_vs_model_b_score_1(platform:str, training_data_source_a:str, training_data_source_b:str, test_data_source:str) -> str:
    """ Constructs a title for a diagram comparing the QWK (Quadratic Weighted Kappa) of score 1 
    from two different models based on their training and test data sources

    Args:
        platform (str): The platform of the models, such as 'BERT' or 'XG-Boost'
        training_data_source_a (str): Data source used to train model A. (e.g., davinci, turbo, gpt4, experts)
        training_data_source_b (str): Data source used to train model B. (e.g., davinci, turbo, gpt4, experts)
        test_data_source (str): Data source on which both models were tested. (e.g., davinci, turbo, gpt4, experts)

    Returns:
        str: The constructed title for the diagram

    """
    return f"QWK of score_1 assigned to samples created by {test_data_source} by two {platform} Models\nOne trained with samples created by {training_data_source_a} one trained with samples created by {training_data_source_b}.\n"


def _get_diagram_title(platform:str, training_data_source: str, test_data_source: str)-> str:
    """ Constructs a title for a diagram displaying results from a single test execution of a model.

    Args:
        platform (str): The platform of the model, such as 'BERT' or 'XG-Boost'
        training_data_source (str): Data source used for training the model (e.g., davinci, turbo, gpt4, experts)
        test_data_source (str): Data source on which the model was tested. This may contain additional info separated by '-'
                                For example, 'davinci-training' suggests data from 'davinci' used for 'training'

    Returns:
        str: The constructed title for the diagram
    """
    # "davinci-training" -> ["davinci", "training"]
    test_data_info = test_data_source.split("-")
    return f"{platform} Model trained with samples created by {_map_source_to_real_name(training_data_source)} rating answers created by {_map_source_to_real_name(test_data_info[0])} used for {test_data_info[1]}"


def _map_source_to_real_name(data_source:str) -> str:
    """ Maps a given data source identifier to its more detailed or official name

    This function provides a mechanism to convert short identifiers or codes used for data sources 
    into their full or real-world names, which might be used in presentations, diagrams, or other contexts
    where clarity and detail are required

    Args:
        data_source (str): Identifier for the data source. Valid values include 'davinci', 'turbo', 'gpt4', and 'experts'

    Returns:
        str: The detailed or official name corresponding to the provided data source identifier

    Raises:
        ValueError: If the provided data source identifier is not recognized
    """
    mapping = {
        "davinci": "text-davinci-003",
        "gpt4": "gpt4",
        "turbo": "text-davinci-003 annotated by gpt-3.5-turbo",
        "experts": "human experts"
    }
    if data_source not in mapping:
        raise ValueError(f"Unknown data source identifier: {data_source}")

    return mapping[data_source]


def _get_figure_for_dataset(diagrams:defaultdict, batch_id:str, training_data_source:str, test_data_source:str, platform:str, diagram_title:str=None) -> KappaFigure:
    """ Retrieves or creates a diagram identified by the specified parameters

    Given a set of parameters, this function will attempt to find a corresponding diagram in the provided `diagrams` dictionary.
    If the diagram does not already exist, a new one will be created with the provided `diagram_title`

    Args:
        diagrams (defaultdict): A dictionary of all existing diagrams
        batch_id (str): Variant identifier
        training_data_source (str): Data source used for training the model (e.g., davinci, turbo, gpt4, experts)
        test_data_source (str): Source of the data the model was tested with (e.g., davinci, turbo, gpt4, experts)
        platform (str): The platform of the model being represented, e.g., 'BERT' or 'XG-Boost'
        diagram_title (str, optional): Title to assign to the diagram if it needs to be created. If a diagram is found, 
            this parameter is ignored. Defaults to None

    Returns:
        KappaFigure: The retrieved or newly created diagram corresponding to the given parameters
    """
    fileLabel = f"{platform}_{training_data_source}_rated_{test_data_source}_samples_{batch_id}"

    if (platform not in diagrams 
        or training_data_source not in diagrams[platform] 
        or test_data_source not in diagrams[platform][training_data_source]
        or batch_id not in diagrams[platform][training_data_source][test_data_source]):
        
        if not diagram_title:
            diagram_title = _get_diagram_title(platform, training_data_source, test_data_source)
        diagrams[platform][training_data_source][test_data_source][batch_id] = KappaFigure(diagram_title, fileLabel)
    
    return diagrams[platform][training_data_source][test_data_source][batch_id]


def _get_scores(answers:List[Answer], score_type:int) -> List[int]:
    """ Retrieves scores of given answers based on the specified score type

    This function extracts scores from a list of Answer objects based on the desired score type specified by the user.
    The score type is identified by an integer (e.g., `score_1`, `score_2`, etc.), and the function will return a list 
    of scores corresponding to that type for all provided answers

    Args:
        answers (List[Answer]): A list of Answer objects from which scores need to be extracted
        score_type (int): An integer representing the desired score type to retrieve (either 1 or 2)

    Returns:
        List[int]: A list of scores corresponding to the specified score type from the provided answers
    """
    return [getattr(answer, f'score_{score_type}') for answer in answers]


def _get_common_answers(answers_a:List[Answer], answers_b:List[Answer]) -> Tuple[List[Answer], List[Answer]]:
    """ Retrieves common answers from two lists based on their answer Ids

    This function compares the answer Ids of two provided lists of answers 
    and returns the common answers from both lists. If an answer Id is found 
    in both lists, the corresponding answer objects from each list are added 
    to the result

    Args:
        answers_a (List[Answer]): The first list of answers to be compared
        answers_b (List[Answer]): The second list of answers to be compared

    Returns:
        Tuple[List[Answer], List[Answer]]: A tuple containing two lists:
            1. List of common answers from the first input list
            2. List of common answers from the second input list
    """
    answer_ids_a = set([answer.answer_id for answer in answers_a])
    answer_ids_b = set([answer.answer_id for answer in answers_b])

    common_answers_a = []
    common_answers_b = []
    for id_a in answer_ids_a:
        if id_a in answer_ids_b:
            common_answers_a.append(next((answer for answer in answers_a if answer.answer_id == id_a), None))
            common_answers_b.append(next((answer for answer in answers_b if answer.answer_id == id_a), None))
    
    return common_answers_a, common_answers_b


def _calculate_kappa_for_models(answers_a:List[Answer], answers_b:List[Answer]) -> Tuple[float, int]:
    """ Calculates the Quadratic Weighted Kappa (QWK) between score_1 of two sets of answers

    This function determines the agreement between two sets of ratings, specifically score_1, using the QWK. If the 
    number of ratings is below a threshold or if the ratings are inconsistent, it provides a warning and returns -5 
    for the QWK

    Args:
        answers_a (List[Answer]): A list of Answer objects from which `score_1` will be extracted
        answers_b (List[Answer]): A second list of Answer objects from which `score_1` will be extracted

    Returns:
        Tuple[float, int]: A tuple containing:
            - QWK (float): The calculated Quadratic Weighted Kappa between the two sets of scores. 
                           Returns -5 if ratings are inconsistent or below threshold
            - Number of Answers (int): The number of answers considered for the QWK calculation

    Notes:
        - The function expects both lists of answers to have more than 100 common answers to provide a 
          valid QWK
        - It also assumes that answer Ids in both lists should match for valid comparison
    """
    return _calculate_kappa_for_score_types(answers_a, 1, answers_b, 1)


def _calculate_kappa_for_model(answers:List[Answer]) -> Tuple[float, int]:
    """ Calculates the Quadratic Weighted Kappa (QWK) between `score_1` and `score_2` of a set of answers

    This function computes the agreement between two sets of scores (`score_1` and `score_2`) from the provided answers 
    using the QWK. If the number of ratings is below a threshold, it provides a warning and returns -5 for the QWK

    Args:
        answers (List[Answer]): A list of Answer objects from which `score_1` and `score_2` will be extracted

    Returns:
        Tuple[float, int]: A tuple containing:
            - QWK (float): The calculated Quadratic Weighted Kappa between `score_1` and `score_2`. 
                           Returns -5 if not enough answers are provided
            - Number of Answers (int): The number of answers considered for the QWK calculation

    Notes:
        - The function expects more than 100 answers to provide a valid QWK calculation
    """
    if len(answers) < 100:
        log.warning(f"Not enough scores: {len(answers)} to calculate kappa")
        return -5, len(answers)

    scores_a = _get_scores(answers, 1)
    scores_b = _get_scores(answers, 2)

    kappa = cohen_kappa_score(scores_a, scores_b, weights='quadratic'), len(scores_a)
    log.debug(f"Calculated QWK: {kappa}")
    return kappa


def _calculate_kappa_for_score_types(answers_a:List[Answer], score_type_a:int, answers_b:List[Answer], score_type_b:int) -> Tuple[float, int]:
    """ Calculates the Quadratic Weighted Kappa (QWK) between specified scores of two sets of answers.

    This function computes the agreement between two sets of scores extracted from the provided answers 
    based on specified score types using the QWK. If the number of ratings in `answers_a` is below a threshold, 
    it logs a warning and returns -5 for the QWK. If the lengths of `answers_a` and `answers_b` are different, 
    it logs an error

    Args:
        answers_a (List[Answer]): List of Answer objects from which scores will be extracted based on `score_type_a`
        score_type_a (int): Type of score to be extracted from `answers_a` (either 1 or 2)
        answers_b (List[Answer]): List of Answer objects from which scores will be extracted based on `score_type_b`
        score_type_b (int): Type of score to be extracted from `answers_b` (either 1 or 2)

    Returns:
        Tuple[float, int]: A tuple containing:
            - QWK (float): The calculated Quadratic Weighted Kappa between the selected scores. 
                            Returns -5 if not enough answers are provided in `answers_a`
            - Number of Answers (int): The number of answers from `answers_a` considered for the QWK calculation

    Notes:
        - The function expects `answers_a` to have more than 100 answers to provide a valid QWK calculation
        - The function expects `answers_a` and `answers_b` to be of the same length for a valid comparison
    """
    common_answers_a, common_answers_b = _get_common_answers(answers_a, answers_b)

    if len(common_answers_a) < 100 or len(common_answers_b) < 100:
        log.warning(f"Not enough common samples: {len(common_answers_a)} from a total of: {len(answers_a)} and {len(common_answers_b)} from a total of: {len(answers_b)}")
        return -5, len(answers_a)

    scores_a = _get_scores(common_answers_a, score_type_a)
    scores_b = _get_scores(common_answers_b, score_type_b)

    return cohen_kappa_score(scores_a, scores_b, weights='quadratic'), len(scores_a)


def _get_min_kappa_count(x_y_boxplot_data_sorted:Dict[str, List[float]]) -> Tuple[int, bool]:
    """ Computes the smallest count of QWK values per sample size in the provided dataset and checks if 
    the count of QWK values is consistent across all datasets

    This function determines the minimum count of QWK values (y-values) corresponding to each sample size 
    (x-values) from the input dataset. It further checks if this count is consistent across all datasets. 
    Entries with only 0 or 1 QWK value(s) are ignored as they don't represent meaningful boxes in the plot

    Args:
        x_y_boxplot_data_sorted (Dict[str, List[float]]): A dictionary where each key represents a sample size 
                                                         and its associated value is a list of QWK values for 
                                                         that sample size

    Returns:
        Tuple[int, bool]: A tuple containing:
            - Minimum QWK Count (int): The smallest number of QWK values present for a sample size. 
                                       Returns -1 if no valid sample sizes are found
            - Consistency Flag (bool): True if the number of QWK values is consistent across all sample sizes, 
                                       False otherwise

    Notes:
        - This function is primarily used for generating informative lines and doesn't impact the actual diagram
        - Boxes representing either 0 or 1 QWK values are disregarded due to their lack of visual significance 
          in the diagram
    """
    # y of 0 will not be rendered as box, y of 1 will reduce a box to a single line
    # both will not just be visible within the diagram it also hardly qualifies as box so we ignore it
    kappa_counts = [len(y) for x, y in x_y_boxplot_data_sorted.items() if len(y) > 1]
    
    if not kappa_counts:
        return -1, True

    min_kappa_count = min(kappa_counts)
    all_equal = all(kappa == min_kappa_count for kappa in kappa_counts)

    for x, y in x_y_boxplot_data_sorted.items():
        log.debug(f"plotting box for sorted x:{x}; y:{y}")
    
    return min_kappa_count, all_equal


def _print_dual_dataset_boxplot(fileLabel:str, x_y_model_data_1:defaultdict[str, List[float]], data_source_1:str, x_y_model_data_2:defaultdict[str, List[float]], data_source_2:str, title:str) -> None:
    """ Generates a combined boxplot from two datasets and saves the diagram as a PDF

    This function visualizes two sets of data in a combined boxplot, where each dataset is represented by
    colored boxes. The combined boxplot is useful for visually comparing two datasets side by side

    Args:
        fileLabel (str): The label that will be used as the filename for the saved boxplot
        x_y_model_data_1 (defaultdict[str, List[float]]): The first dataset to be plotted, with x-values 
                                                          (keys) mapping to a list of y-values (QWKs)
        data_source_1 (str): Data source of x_y_model_data_1
        x_y_model_data_2 (defaultdict[str, List[float]]): The second dataset to be plotted, with x-values 
                                                          (keys) mapping to a list of y-values (QWKs)
        data_source_2 (str): Data source of x_y_model_data_2
        title (str): The title to be displayed on the generated boxplot

    Notes:
        - The boxplot is saved as a PDF file in the location specified by the configuration
        - The first dataset's boxes are colored red, while the second dataset's boxes are colored blue
        - Boxplots are sorted in ascending order of their x-values
    """
    figure, ax = plt.subplots()

    for key in set(x_y_model_data_1.keys()).union(x_y_model_data_2.keys()):
        if key not in x_y_model_data_1:
            x_y_model_data_1[key] = []
        if key not in x_y_model_data_2:
            x_y_model_data_2[key] = []

    # sort x ascending
    x_y_dataset_1_sorted = dict(sorted(x_y_model_data_1.items()))
    x_y_dataset_2_sorted = dict(sorted(x_y_model_data_2.items()))

    min_kappa_count_dataset_1, for_all_boxes_equal_d1 = _get_min_kappa_count(x_y_dataset_1_sorted)
    min_kappa_count_dataset_2, for_all_boxes_equal_d2 = _get_min_kappa_count(x_y_dataset_2_sorted)

    x_values_sorted = list(x_y_dataset_1_sorted.keys())

    positions_1 = [i - 0.2 for i, key in enumerate(x_y_dataset_1_sorted.keys())]
    positions_2 = [i + 0.2 for i, key in enumerate(x_y_dataset_1_sorted.keys())]

    log.debug(f"About to plot dataset_1: {len(x_y_dataset_1_sorted)} with: {len(positions_1)} positions")
    log.debug(f"About to plot dataset_2: {len(x_y_dataset_2_sorted)} with: {len(positions_2)} positions")
    boxes_1 = ax.boxplot([sorted(data) for data in x_y_dataset_1_sorted.values()], positions=positions_1, widths=0.4, patch_artist=True)
    boxes_2 = ax.boxplot([sorted(data) for data in x_y_dataset_2_sorted.values()], positions=positions_2, widths=0.4, patch_artist=True)

    for bplot, color in zip((boxes_1, boxes_2), ["red", "blue"]):
        for box in bplot['boxes']:
            box.set_facecolor(color)

    plt.xticks([i for i in range(len(x_values_sorted))], x_values_sorted)

    if for_all_boxes_equal_d1 and for_all_boxes_equal_d2 and min_kappa_count_dataset_1 == min_kappa_count_dataset_2:
        figtext = f"Each box represents exactly {min_kappa_count_dataset_1}"
    else:
        figtext = f"Each box represents at leased {min(min_kappa_count_dataset_1, min_kappa_count_dataset_2)}"
    
    figtext += f" kappa values\n{data_source_1} is represented by red\n{data_source_2} is represented by blue"
    _add_figtext(figure, figtext)

    ax.set_title(title)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('QWK')

    figure.savefig(config.get_path_for_qwk_file(f"boxplot_kappa_{fileLabel}.pdf"), bbox_inches='tight')


def _print_boxplot(identifier:str, x_y_boxplot_data:Dict[str, List[float]], title:str) -> None:
    """ Generates a boxplot from the provided dataset and saves the diagram as a PDF

    This function visualizes the provided dataset in a boxplot, which is useful for
    visually representing the distribution and variability of the dataset

    Args:
        identifier (str): A unique identifier used to create the filename for the saved boxplot
        x_y_boxplot_data (Dict[str, List[float]]): The dataset to be plotted, where x-values (keys)
                                                   map to a list of y-values (QWKs)
        title (str): The title to be displayed on the generated boxplot

    Notes:
        - The boxplot is saved as a PDF file in the location specified by the configuration
        - Boxplots are sorted in ascending order of their x-values
    """
    figure, ax = plt.subplots()

    # sort x ascending
    x_y_boxplot_data_sorted = dict(sorted(x_y_boxplot_data.items()))

    # sort each individual y value ascending
    ax.boxplot([sorted(y) for y in x_y_boxplot_data_sorted.values()])
    ax.set_xticklabels(list(x_y_boxplot_data_sorted.keys()))

    min_kappa_count, for_all_boxes_equal = _get_min_kappa_count(x_y_boxplot_data_sorted)
    if for_all_boxes_equal:
        _add_figtext(figure, f"Each box represents exactly {min_kappa_count} kappa values")
    else:
        _add_figtext(figure, f"Each box represents at leased {min_kappa_count} kappa values")

    ax.set_title(title)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('QWK')

    figure.savefig(config.get_path_for_qwk_file(f"boxplot_kappa_{identifier}.pdf"), bbox_inches='tight')


def _diagram_gpt_vs_experts(chat_gpt_model_type:str) -> None:
    """ Compares QWK data from a specific ChatGPT model with data created by human experts
    
    This function visualizes the QWK scores, comparing all available combinations of score types
    between the selected ChatGPT model and the data created by human experts.
    It also saves the resulting diagram
    
    Args:
        chat_gpt_model_type (str): The type of ChatGPT model for which the data should be compared (either, davinci, turbo or gpt4)
    
    Notes:
        - The function loads datasets for both the ChatGPT model and human experts
        - It generates a diagram visualizing the QWK scores for various combinations of scores
        - The resulting diagram is saved using the specified configuration
    """
    chat_gpt_sample_path = config.get_samples_path(chat_gpt_model_type)
    if not os.path.exists(chat_gpt_sample_path):
        log.info(f"The file {chat_gpt_sample_path} does not exist")
        return

    all_ai_answers = get_answers(chat_gpt_sample_path)
    all_expert_answers = get_answers(config.get_samples_path("experts"))

    chat_gpt_rating_expert_data = get_answers(config.get_samples_path(f"{chat_gpt_model_type}_rating_expert_data"))

    chat_gpt_vs_experts = KappaFigure(f"Kappa of ChatGPT {chat_gpt_model_type} vs Experts", f"chat_gpt_{chat_gpt_model_type}_vs_experts")
    chat_gpt_vs_experts.plot("GPT-Dataset score_1 vs score_2", *_calculate_kappa_for_model(all_ai_answers))
    chat_gpt_vs_experts.plot("Expert-Dataset score_1 vs score_2", *_calculate_kappa_for_model(all_expert_answers))
    chat_gpt_vs_experts.plot("GPT-Rating-Expert-Dataset  score_1 vs score_2", *_calculate_kappa_for_model(chat_gpt_rating_expert_data))
    chat_gpt_vs_experts.plot("GPT-VS-Expert score_1 vs score_1", *_calculate_kappa_for_score_types(chat_gpt_rating_expert_data, 1, all_expert_answers, 1))
    chat_gpt_vs_experts.plot("GPT-VS-Expert score_1 vs score_2", *_calculate_kappa_for_score_types(chat_gpt_rating_expert_data, 1, all_expert_answers, 2))
    chat_gpt_vs_experts.plot("GPT-VS-Expert score_2 vs score_1", *_calculate_kappa_for_score_types(chat_gpt_rating_expert_data, 2, all_expert_answers, 1))
    chat_gpt_vs_experts.plot("GPT-VS-Expert score_2 vs score_2", *_calculate_kappa_for_score_types(chat_gpt_rating_expert_data, 2, all_expert_answers, 2))
    chat_gpt_vs_experts.save(config, f"The ChatGPT {chat_gpt_model_type}-Dataset consists of {len(all_ai_answers)} samples.\n The Expert-Dataset consists of {len(all_expert_answers)} samples.", True)


def _allocate_results(result_path:str, training_data_source:str, test_data_source:str, test_result_sets:defaultdict, batch_size:int, batch_id:str, model_type:str) -> None:
    """ Allocate given results for later analysis, calculates the QWK between `score_1` and `score_2`, 
    and produces a corresponding diagram
    
    This function evaluates the results from a specified path, processes them, 
    and plots the QWK values for the test results. It further segregates test results 
    based on their testing status and updates the provided test result sets accordingly
    
    Args:
        result_path (str): Path to the results file.
        training_data_source (str): Data source used for training the model (e.g., davinci, turbo, gpt4, experts)
        test_data_source (str): Source of the data the model was tested with (e.g., davinci, turbo, gpt4, experts)
        test_result_sets (defaultdict): Dictionary that holds answers, which are updated and used for later processing
        batch_size (int): Number of answers the model was trained with
        batch_id (str): Variant identifier
        model_type (str): Type of the model trained ('BERT' or 'XG-Boost')
    
    Notes:
        - Data used for testing is considered valid for comparison across models
        - Data used for training is not compared across different models
    """
    if not os.path.exists(result_path):
        log.info(f"The file {result_path} does not exist")
        return
    log.debug(f"Load results for file: {result_path}")

    test_results = get_answers(result_path)

    # we do not compare data used for training across models
    if test_data_source.endswith("-testing") or test_data_source == "experts-training":
        _add_list_entry(test_result_sets[model_type], test_data_source, {
            "results": test_results,
            "training_data_source": training_data_source
        })
    log.debug(f"Calculate kappa for: {result_path}")
    _get_figure_for_dataset(diagrams, batch_id, training_data_source, test_data_source, model_type.upper()).plot(batch_size, *_calculate_kappa_for_model(test_results))


def _cleanup(config:Configuration) -> None:
    """ Ensures that none of the files, created by this module, already exist

    Args:
        config (Configuration): Allows access to the projects central configuration
    """
    base_folder = config.get_qwk_root_path()
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)
    
    os.makedirs(base_folder, exist_ok=True)


if __name__ == "__main__":
    config_logger(log.DEBUG, "calculate_kappa.log")
    config = Configuration()
    _cleanup(config)

    trained_model_types = ['bert', 'xgb']
    data_sources = ['experts', 'davinci', 'turbo', 'gpt4']

    diagrams = _recursive_default_dict()
    for batch in config.get_batches():
        for batch_id in batch.ids:
            log.debug(f"Calculate graph for batch size: {batch.size} and id: {batch_id}")
            test_result_sets = _recursive_default_dict()
            
            for model_type in trained_model_types:

                # each model rates its own training and test data sources and the test data sources of all other models
                for data_source_a in data_sources:
                    result_training_path = config.get_test_results_path(model_type, data_source_a, f"{data_source_a}-training", batch.size, batch_id)
                    _allocate_results(result_training_path, data_source_a, f"{data_source_a}-training", test_result_sets, batch.size, batch_id, model_type)

                    if data_source_a != "experts":
                        result_training_path = config.get_test_results_path(model_type, data_source_a, "experts-training", batch.size, batch_id)
                        _allocate_results(result_training_path, data_source_a, "experts-training", test_result_sets, batch.size, batch_id, model_type)
                    
                    for data_source_b in data_sources:
                        result_path = config.get_test_results_path(model_type, data_source_a, f"{data_source_b}-testing", batch.size, batch_id)
                        _allocate_results(result_path, data_source_a, f"{data_source_b}-testing", test_result_sets, batch.size, batch_id, model_type)

            # calculate kappa of score_1 vs score_1 of all test data result sets
            for model_type, sub_dict in test_result_sets.items():
                platform = model_type.upper()
                for test_data_source, result_sets in sub_dict.items():
                    for result_set_a in result_sets:
                        training_data_source_a = result_set_a["training_data_source"]
                        results_a = result_set_a["results"]

                        for result_set_b in result_sets:
                            # Only compare different result sets
                            if result_set_a == result_set_b:
                                continue
    
                            training_data_source_b = result_set_b["training_data_source"]
                            results_b = result_set_b["results"]

                            descriptor = f"{training_data_source_a}_vs_{training_data_source_b}_score_1"
                            title = _get_diagram_title_model_a_score_1_vs_model_b_score_1(platform, training_data_source_a, training_data_source_b, test_data_source)
                            kappa, answer_count = _calculate_kappa_for_models(results_a, results_b)
                            log.debug(f"{platform}_{descriptor} rated {test_data_source}: - batch: {batch.size} kappa: {kappa} samples: {answer_count}")
                            if kappa >= -1 and kappa <= 1:
                                _get_figure_for_dataset(diagrams, batch_id, descriptor, test_data_source, platform, title).plot(batch.size, kappa, answer_count)
                            else:
                                log.error(f"Kappa is out of bound: {kappa}")

    # ChatGPT rated the samples created by human experts, but not the other way around
    _diagram_gpt_vs_experts("davinci")
    _diagram_gpt_vs_experts("turbo")
    _diagram_gpt_vs_experts("gpt4")

    # sort data for boxplots
    accumulated_data = _recursive_default_dict()
    for platform, sub_dict_a in diagrams.items():
        for training_data_source, sub_dict_b in sub_dict_a.items():
            for test_data_source, sub_dict_c in sub_dict_b.items():

                # x and y for all variants
                x_y_boxplot_data = _recursive_default_dict()
                for batch_id, diagram in sub_dict_c.items():

                    # all data points for this diagram
                    x_batch_sizes, y_kappas = diagram.get_values()

                    log.debug(f"Adding x: {x_batch_sizes}, y: {y_kappas} to boxplot matrix")
                    assert len(x_batch_sizes) == len(y_kappas), f"Found inconsistent data for diagram: {training_data_source} {test_data_source} {batch_id}; {len(x_batch_sizes)} != {len(y_kappas)}"

                    for i, batch_size in enumerate(x_batch_sizes):
                        _add_list_entry(x_y_boxplot_data, batch_size, y_kappas[i])

                    # print diagram
                    diagram.save(config)
                
                # If we compare different models, both sources are combined
                if "_vs_" in training_data_source:
                    sources_str = re.sub('_score_\d+', '', training_data_source)
                    training_data_sources = sources_str.split("_vs_")
                    title = _get_diagram_title_model_a_score_1_vs_model_b_score_1(platform, training_data_sources[0], training_data_sources[1], test_data_source)
                else:
                    title = _get_diagram_title(platform, training_data_source, test_data_source)

                _print_boxplot(f"{platform}_{training_data_source}_model_{test_data_source}", x_y_boxplot_data, title)

                # collect all boxplots displaing the rating of data that was reserved for tests
                if test_data_source.endswith("-testing") or test_data_source == "experts-training":
                    _add_list_entry(accumulated_data[platform], test_data_source, {
                        "training_data_source": training_data_source,
                        "x_y_boxplot_data": x_y_boxplot_data
                    })

    # boxplots with two datasets
    for platform,  sub_dict_a in accumulated_data.items():
        for test_data_source, result_sets in sub_dict_a.items():
            test_data_source_split = test_data_source.split("-")

            for result_set_a in result_sets:
                training_data_source_a = result_set_a["training_data_source"]
                x_y_boxplot_data_a = result_set_a["x_y_boxplot_data"]
                
                for result_set_b in result_sets:

                    if result_set_a == result_set_b:
                        continue

                    training_data_source_b = result_set_b["training_data_source"]
                    x_y_boxplot_data_b = result_set_b["x_y_boxplot_data"]

                    if "_vs_" in training_data_source_a:
                        data_source_a = re.sub('_score_\d+', '', training_data_source_a)
                        data_sources_a = data_source_a.split("_vs_")
                        file_name_segment_a = data_source_a
                        # score_1 of model A vs score_1 of model B compared to score_1 of model C vs score_1 of model D
                        # this shows which model rate more similar
                        comparison_type = "model_vs_model_compared_to_model_vs_model"
                    else:
                        data_source_a = _map_source_to_real_name(training_data_source_a)
                        file_name_segment_a = training_data_source_a
                        # model A vs the data_source of the rated data (gpt4, experts ect.) compared to model B vs the data_source of the rated data (gpt4, experts ect.)
                        comparison_type = "model_vs_model"
                    
                    if "_vs_" in training_data_source_b:
                        # It doesn't make sense to compare the QWK of one trained model with its data source to the QWK between two other models
                        if comparison_type == "model_vs_model":
                            continue

                        data_source_b = re.sub('_score_\d+', '', training_data_source_b)
                        data_sources_b = data_source_b.split("_vs_")
                        file_name_segment_b = data_source_b

                        title = f"""The QWK for score_1 when comparing two {platform} models versus the QWK for score_1 between a different pair of {platform} models.
The first pair was trained with samples created by {_map_source_to_real_name(data_sources_a[0])} and {_map_source_to_real_name(data_sources_a[1])},
the second pair was trained with samples created by {_map_source_to_real_name(data_sources_b[0])} and {_map_source_to_real_name(data_sources_b[1])}"""

                    else:
                        # It doesn't make sense to compare the QWK of one trained model with its data source to the QWK between two other models
                        if comparison_type == "model_vs_model_compared_to_model_vs_model":
                            continue

                        data_source_b = _map_source_to_real_name(training_data_source_b)
                        file_name_segment_b = training_data_source_b

                        title = f"""QWK of two {platform} Models with the source of the data. One trained with samples created by {data_source_a} One trained with samples created by {data_source_b}.
Both rated answers created by {_map_source_to_real_name(test_data_source_split[0])} used for {test_data_source_split[1]}"""

                    fileLabel = f"{platform}_{file_name_segment_a}_compared_to_{file_name_segment_b}_score_1_rated_{test_data_source}_samples"
                    _print_dual_dataset_boxplot(fileLabel, x_y_boxplot_data_a, data_source_a, x_y_boxplot_data_b, data_source_b, title)
