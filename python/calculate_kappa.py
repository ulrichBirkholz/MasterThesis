from config import Configuration
from tsv_utils import get_results, TestResult
from typing import List, Tuple, Dict, Any
from tsv_utils import get_answers
import logging as log
from config_logger import config_logger

from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import chain
import os


class KappaFigure():
    def __init__(self, title:str, fileLabel:str) -> None:
        self.answer_count = -1
        self.all_counts_equal = True

        self.filename = f"kappa_{fileLabel}.pdf"
        self.title = title

        self.kappa_values = []
        self.labels = []
    

    def get_values(self):
        return self.labels, self.kappa_values


    def plot(self, label, kappa, answer_count):
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


    def save(self, config: Configuration, figtext:str=None, compact:bool=False):
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
        figure.savefig(config.get_path_for_datafile(self.filename), bbox_inches='tight')


def _add_figtext(fig, figtext):
    fig.subplots_adjust(bottom=0.2)

    fig.text(0.5, 0.1, '_'*80, ha='center')
    fig.text(0.5, 0.08, figtext, ha='center', va='top')


def _recursive_default_dict():
    return defaultdict(_recursive_default_dict)


def _add_list_entry(dict:defaultdict, name:str, entry:Any) -> None:
    if name in dict:
        dict[name].append(entry)
    else:
        dict[name] = [entry]


def _get_diagram_title_model_vs_model(platform:str, training_data_source_a:str, training_data_source_b:str, test_data_source:str):
    return f"QWK of score_1 assigned to samples created by {test_data_source} by two {platform} Models\nOne trained with samples created by {training_data_source_a} one trained with samples created by {training_data_source_b}.\n"


def _map_source_to_real_name(data_source:str) -> str:
    if data_source == "davinci":
        return "text-davinci-003"
    if data_source == "gpt4":
        return "gpt4"
    if data_source == "turbo":
        return "text-davinci-003 annotated by gpt-3.5-turbo"
    if data_source == "expert":
        return "human Experts"


def _get_diagram_title(platform:str, training_data_source: str, test_data_source: str)-> str:
    # "davinci-training" -> ["davinci", "training"]
    test_data_info = test_data_source.split("-")
    return f"{platform} Model trained with samples created by {_map_source_to_real_name(training_data_source)} rating answers created by {_map_source_to_real_name(test_data_info[0])} used for {test_data_info[1]}"


def _get_figure_for_dataset(diagrams:defaultdict, batch_id:str, training_data_source:str, test_data_source:str, platform:str, diagram_title:str=None) -> KappaFigure:
    fileLabel = f"{platform}_{training_data_source}_rated_{test_data_source}_samples_{batch_id}"

    if (platform not in diagrams 
        or training_data_source not in diagrams[platform] 
        or test_data_source not in diagrams[platform][training_data_source]
        or batch_id not in diagrams[platform][training_data_source][test_data_source]):
        
        if not diagram_title:
            diagram_title = _get_diagram_title(platform, training_data_source, test_data_source)
        diagrams[platform][training_data_source][test_data_source][batch_id] = KappaFigure(diagram_title, fileLabel)
    
    return diagrams[platform][training_data_source][test_data_source][batch_id]


def _get_scores(ratings: List[TestResult], score_type:int) -> List[int]:
    return [getattr(rating.answer, f'score_{score_type}') for rating in ratings]


def _calculate_kappa_for_models(ratings_a:List[TestResult], ratings_b:List[TestResult]):
    if len(ratings_a) != len(ratings_b) or len(ratings_a) < 100:
        log.warning(f"Not enough samples or ratings are inconsistent: {len(ratings_a)} vs. {len(ratings_b)}")
        return -5, len(ratings_a)
    
    rating_ids_a = set([rating.answer.answer_id for rating in ratings_a])
    rating_ids_b = set([rating.answer.answer_id for rating in ratings_b])

    common_ids = rating_ids_a.intersection(rating_ids_b)
    if len(rating_ids_a) != len(common_ids):
        a_set = set(rating_ids_a)
        b_set = set(rating_ids_b)

        a_diff = [item for item in a_set if item not in common_ids]
        b_diff = [item for item in b_set if item not in common_ids]

        log.warning(f"Ratings are inconsistent: {len(ratings_a)} vs. {len(ratings_b)}, common_ids: {len(common_ids)}")
        log.warning(f"diff ratings_a to common: {a_diff}, diff ratings_b to common: {b_diff}")
        return -5, len(ratings_a)

    scores_a = _get_scores(ratings_a, 1)
    scores_b = _get_scores(ratings_b, 1)

    return cohen_kappa_score(scores_a, scores_b, weights='quadratic'), len(scores_a)


def _calculate_kappa_for_model(ratings:List[TestResult]):
    if len(ratings) < 100:
        log.warning(f"Not enough scores: {len(ratings)} to calculate kappa")
        return -5, len(ratings)

    scores_a = _get_scores(ratings, 1)
    scores_b = _get_scores(ratings, 2)

    return cohen_kappa_score(scores_a, scores_b, weights='quadratic'), len(scores_a)

def _calculate_kappa_for_score_types(ratings_a:List[TestResult], score_type_a, ratings_b:List[TestResult], score_type_b):
    if len(ratings_a) < 100:
        log.warning(f"Not enough scores: {len(ratings_a)} to calculate kappa")
        return -5, len(ratings_a)
    
    if len(ratings_a) != len(ratings_b):
        log.error(f"We try to compare ratings of different length, a: {len(ratings_a)}, b: {len(ratings_b)}")

    scores_a = _get_scores(ratings_a, score_type_a)
    scores_b = _get_scores(ratings_b, score_type_b)

    return cohen_kappa_score(scores_a, scores_b, weights='quadratic'), len(scores_a)


def _answers_to_rating(answers):
     return [TestResult(None, answer) for answer in answers]

# Just extracts data for the info line, this does not affect the diagram
def _get_min_kappa_count(x_y_boxplot_data_sorted):
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


def _print_dual_dataset_boxplot(fileLabel:str, x_y_model_data_1, x_y_model_data_2, title:str):
    figure, ax = plt.subplots()

    x_y_dataset_1 = x_y_model_data_1[0]
    x_y_dataset_2 = x_y_model_data_2[0]
    for key in set(x_y_dataset_1.keys()).union(x_y_dataset_2.keys()):
        if key not in x_y_dataset_1:
            x_y_dataset_1[key] = []
        if key not in x_y_dataset_2:
            x_y_dataset_2[key] = []

    # sort x ascending
    x_y_dataset_1_sorted = dict(sorted(x_y_dataset_1.items()))
    x_y_dataset_2_sorted = dict(sorted(x_y_dataset_2.items()))

    min_kappa_count_dataset_1, for_all_boxes_equal_d1 = _get_min_kappa_count(x_y_dataset_1_sorted)
    min_kappa_count_dataset_2, for_all_boxes_equal_d2 = _get_min_kappa_count(x_y_dataset_2_sorted)

    x_values_sorted = list(x_y_dataset_1_sorted.keys())

    positions_1 = [i - 0.2 for i, key in enumerate(x_y_dataset_1_sorted.keys())]
    positions_2 = [i + 0.2 for i, key in enumerate(x_y_dataset_1_sorted.keys())]

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
    
    figtext += f" kappa values\n{x_y_model_data_1[1]} is represented by red\n{x_y_model_data_2[1]} is represented by blue"
    _add_figtext(figure, figtext)

    ax.set_title(title)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('QWK')

    figure.savefig(config.get_path_for_datafile(f"boxplot_kappa_{fileLabel}.pdf"), bbox_inches='tight')


def _print_boxplot(identifier:str, x_y_boxplot_data, title:str) -> None:
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

    figure.savefig(config.get_path_for_datafile(f"boxplot_kappa_{identifier}.pdf"), bbox_inches='tight')


def _diagram_gpt_vs_experts(chat_gpt_model_type:str):

        chat_gpt_sample_path = config.get_samples_path(chat_gpt_model_type)
        if not os.path.exists(chat_gpt_sample_path):
            log.info(f"The file {chat_gpt_sample_path} does not exist")
            return

        all_ai_answers = _answers_to_rating(get_answers(chat_gpt_sample_path))
        all_expert_answers = _answers_to_rating(get_answers(config.get_samples_path("experts")))

        chat_gpt_rating_expert_data = _answers_to_rating(get_answers(config.get_samples_path(f"{chat_gpt_model_type}_rating_expert_data")))

        chat_gpt_vs_experts = KappaFigure(f"Kappa of ChatGPT {chat_gpt_model_type} vs Experts", f"chat_gpt_{chat_gpt_model_type}_vs_experts")
        chat_gpt_vs_experts.plot("GPT-Dataset score_1 vs score_2", *_calculate_kappa_for_model(all_ai_answers))
        chat_gpt_vs_experts.plot("Expert-Dataset score_1 vs score_2", *_calculate_kappa_for_model(all_expert_answers))
        chat_gpt_vs_experts.plot("GPT-Rating-Expert-Dataset  score_1 vs score_2", *_calculate_kappa_for_model(chat_gpt_rating_expert_data))
        chat_gpt_vs_experts.plot("GPT-VS-Expert score_1 vs score_1", *_calculate_kappa_for_score_types(chat_gpt_rating_expert_data, 1, all_expert_answers, 1))
        chat_gpt_vs_experts.plot("GPT-VS-Expert score_1 vs score_2", *_calculate_kappa_for_score_types(chat_gpt_rating_expert_data, 1, all_expert_answers, 2))
        chat_gpt_vs_experts.plot("GPT-VS-Expert score_2 vs score_1", *_calculate_kappa_for_score_types(chat_gpt_rating_expert_data, 2, all_expert_answers, 1))
        chat_gpt_vs_experts.plot("GPT-VS-Expert score_2 vs score_2", *_calculate_kappa_for_score_types(chat_gpt_rating_expert_data, 2, all_expert_answers, 2))
        chat_gpt_vs_experts.save(config, f"The ChatGPT {chat_gpt_model_type}-Dataset consists of {len(all_ai_answers)} samples.\n The Expert-Dataset consists of {len(all_expert_answers)} samples.", True)

def _allocate_results(result_path:str, questions_path:str, training_data_source:str, test_data_source:str, test_result_sets:defaultdict, batch_id:str, model_type:str):
    if not os.path.exists(result_path):
        log.info(f"The file {result_path} does not exist")
        return

    test_results = get_results(result_path, questions_path)

    # we do not compare data used for training across models
    if test_data_source.endswith("-testing"):
        _add_list_entry(test_result_sets[model_type], test_data_source, {
            "results": test_results,
            "training_data_source": training_data_source
        })

    _get_figure_for_dataset(diagrams, batch_id, training_data_source, test_data_source, model_type.upper()).plot(batch_size.size, *_calculate_kappa_for_model(test_results))

if __name__ == "__main__":
    config_logger(log.DEBUG, "calculate_kappa.log")
    config = Configuration()
    questions_path = config.get_questions_path()
    
    trained_model_types : ['bert', 'xgb']
    data_sources = ['experts', 'davinci', 'turbo', 'gpt4']

    diagrams = _recursive_default_dict()
    for batch_size in config.get_batch_sizes():
        for batch_id in batch_size.ids:
            log.debug(f"Calculate graph for batch size: {batch_size.size} and id: {id}")
            test_result_sets = _recursive_default_dict()
            
            for model_type in trained_model_types:

                # each model rates its own training and test data sources and the test data sources of all other models
                for i in range(len(data_sources)):
                    data_source = data_sources[i]
                    result_training_path = config.get_test_results_path(f"{model_type}_{data_source}", f"{data_source}-training", batch_size, batch_id)
                    _allocate_results(result_training_path, questions_path, data_source, f"{data_source}-training", test_result_sets, batch_id, model_type)
                    
                    for j in range(i, len(data_sources)):
                        data_source_j = data_sources[j]
                        result_path = config.get_test_results_path(f"{model_type}_{data_source}", f"{data_source_j}-testing", batch_size, batch_id)
                        _allocate_results(result_training_path, questions_path, data_source, f"{data_source_j}-testing", test_result_sets, batch_id, model_type)

            # calculate kappa of score_1 vs score_1 of all test data result sets
            for model_type, sub_dict in test_result_sets.items():
                platform = model_type.upper()
                for test_data_source, result_sets in sub_dict.items():
                    for a in range(len(result_sets)):
                        training_data_source_a = result_sets[a]["training_data_source"]
                        result_set_a = result_sets[a]["results"]

                        for b in range(a + 1, len(result_sets)):
                            training_data_source_b = result_sets[b]["training_data_source"]
                            result_set_b = result_sets[b]["results"]

                            descriptor = f"{training_data_source_a}_vs_{training_data_source_b}_score_1"
                            title = _get_diagram_title_model_vs_model(platform, training_data_source_a, training_data_source_b, test_data_source)
                            kappa, answer_count = _calculate_kappa_for_models(result_set_a, result_set_b)
                            log.debug(f"{platform}_{descriptor} rated {test_data_source}: - batch_size: {batch_size.size} kappa: {kappa} samples: {answer_count}")
                            if kappa >= -1 and kappa <= 1:
                                _get_figure_for_dataset(diagrams, batch_id, descriptor, test_data_source, platform, title).plot(batch_size.size, kappa, answer_count)
                            else:
                                log.error(f"Kappa is out of bound: {kappa}")

    _diagram_gpt_vs_experts("davinci")
    _diagram_gpt_vs_experts("turbo")
    _diagram_gpt_vs_experts("gpt4")

    # sort data for boxplots
    accumulated_data = _recursive_default_dict()
    for platform, sub_dict_a in diagrams.items():
        for training_data_source, sub_dict_b in diagrams.items():
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
                
                title = _get_diagram_title(platform, training_data_source, test_data_source)
                _print_boxplot(f"{platform}_{training_data_source}_model_{test_data_source}", x_y_boxplot_data, title)

                # collect all boxplots displaing the rating of data that was reserved for tests
                if test_data_source.endswith("-testing"):
                    _add_list_entry(accumulated_data[platform], test_data_source, {
                        "training_data_source": training_data_source,
                        "x_y_boxplot_data": x_y_boxplot_data
                    })

    # boxplots with two datasets
    for platform,  sub_dict_a in accumulated_data.items():
        for test_data_source, result_sets in sub_dict_a.items():
            for a in range(len(result_sets)):
                training_data_source_a = result_sets[a]["training_data_source"]
                x_y_boxplot_data_a = result_sets[a]["x_y_boxplot_data"]
                
                for b in range(a + 1, len(result_sets)):
                    training_data_source_b = result_sets[b]["training_data_source"]
                    x_y_boxplot_data_b = result_sets[b]["x_y_boxplot_data"]

                    title = _get_diagram_title_model_vs_model(platform, training_data_source_a, training_data_source_b, test_data_source)
                    fileLabel = f"{platform}_{training_data_source_a}_vs_{training_data_source_b}_score_1_rated_{test_data_source}_samples"
                    _print_dual_dataset_boxplot(fileLabel, x_y_boxplot_data_a, x_y_boxplot_data_b, title)
