from config import Configuration
from tsv_utils import get_ratings, Rating
from typing import List, Tuple, Dict
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


def _get_platform_from_model(training_data_source: str):
    return training_data_source.split("_")[0].upper()


def _get_diagram_title(training_data_source: str, test_data_source: str):
    platform = _get_platform_from_model(training_data_source)
    descriptors = {
        "man-training": " ML Model\nrating manually created answers used for training",
        "ai-training": " ML Model\nrating answers created by AI used for training",
        "man-rating": " ML Model\nrating manually created answers",
        "ai-rating": " ML Model\nrating answers created by AI"
    }

    refined_type = "AI-Refined" if training_data_source.endswith("ai") else "Expert-Refined"

    refined_type = ("AI-Refined compared with Expert-Refined" if training_data_source.endswith("ai_v_man") 
                    else "AI-Refined" if training_data_source.endswith("ai") 
                    else "Expert-Refined")

    title = f"{platform} {refined_type}"

    if test_data_source in descriptors:
        title += descriptors[test_data_source]
    else:
        log.warning(f"The test_data_source {test_data_source} is invalid")
        return
    
    return title


def _get_figure_for_dataset(diagrams:Dict, id:str, training_data_source:str, test_data_source:str) -> KappaFigure:
    fileLabel = f"{training_data_source}_model_{test_data_source}_answers_{id}"

    if training_data_source not in diagrams or test_data_source not in diagrams[training_data_source] or id not in diagrams[training_data_source][test_data_source]:
        diagram_title = _get_diagram_title(training_data_source, test_data_source)
        
        if diagram_title is None:
            return
        
        diagrams[training_data_source][test_data_source][id] = KappaFigure(diagram_title, fileLabel)
    
    return diagrams[training_data_source][test_data_source][id]


def _get_scores(ratings: List[Rating], score_type:int) -> List[int]:
    return [getattr(rating.answer, f'score_{score_type}') for rating in ratings]


def _calculate_kappa_for_models(ratings_a:List[Rating], ratings_b:List[Rating]):
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


def _calculate_kappa_for_model(ratings:List[Rating]):
    if len(ratings) < 100:
        log.warning(f"Not enough scores: {len(ratings)} to calculate kappa")
        return -5, len(ratings)

    scores_a = _get_scores(ratings, 1)
    scores_b = _get_scores(ratings, 2)

    return cohen_kappa_score(scores_a, scores_b, weights='quadratic'), len(scores_a)

def _calculate_kappa_for_score_types(ratings_a:List[Rating], score_type_a, ratings_b:List[Rating], score_type_b):
    if len(ratings_a) < 100:
        log.warning(f"Not enough scores: {len(ratings_a)} to calculate kappa")
        return -5, len(ratings_a)
    
    if len(ratings_a) != len(ratings_b):
        log.error(f"We try to compare ratings of different length, a: {len(ratings_a)}, b: {len(ratings_b)}")

    scores_a = _get_scores(ratings_a, score_type_a)
    scores_b = _get_scores(ratings_b, score_type_b)

    return cohen_kappa_score(scores_a, scores_b, weights='quadratic'), len(scores_a)


def _answers_to_rating(answers):
     return [Rating(None, answer) for answer in answers]

def _filter_model_ratings(modle_dict, model_set):
    ratings = []
    for model, rating in modle_dict.items():
        if model in model_set:
            platform = _get_platform_from_model(model)
            ratings.append(rating)
    
    return ratings, platform

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


def _print_dual_dataset_boxplot(identifier: str, x_y_model_data_1, x_y_model_data_2, title: str):
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

    figure.savefig(config.get_path_for_datafile(f"boxplot_kappa_{identifier}.pdf"), bbox_inches='tight')


def _print_boxplot(identifier: str, x_y_boxplot_data, title: str):
    figure, ax = plt.subplots()

    # sort x ascending
    x_y_boxplot_data_sorted = dict(sorted(x_y_boxplot_data.items()))

    min_kappa_count, for_all_boxes_equal = _get_min_kappa_count(x_y_boxplot_data_sorted)

    # sort each individual y value ascending
    ax.boxplot([sorted(y) for y in x_y_boxplot_data_sorted.values()])
    ax.set_xticklabels(list(x_y_boxplot_data_sorted.keys()))

    if for_all_boxes_equal:
        _add_figtext(figure, f"Each box represents exactly {min_kappa_count} kappa values")
    else:
        _add_figtext(figure, f"Each box represents at leased {min_kappa_count} kappa values")

    ax.set_title(title)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('QWK')

    figure.savefig(config.get_path_for_datafile(f"boxplot_kappa_{identifier}.pdf"), bbox_inches='tight')

if __name__ == "__main__":
    config_logger(log.DEBUG, "calculate_kappa.log")
    config = Configuration()
    questions_path = config.get_questions_path()

    
    model_sets = [['bert_ai', 'bert_man'], ['xgb_ai', 'xgb_man']]

    training_data = ['ai-training', 'man-training']
    rating_data = ['ai-rating', 'man-rating']

    diagrams = _recursive_default_dict()
    for batch_size in config.get_batch_sizes():
        for id in batch_size.ids:
            log.debug(f"Calculate graph for batch size: {batch_size.size} and id: {id}")
            rating_sets = _recursive_default_dict()

            for model in chain.from_iterable(model_sets):
                for type in training_data + rating_data:
                    rating_path = config.get_test_results_path(model, type, batch_size.size, id)
                    if not os.path.exists(rating_path):
                        log.info(f"The file {rating_path} does not exist")
                        continue

                    model_ratings = get_ratings(config.get_test_results_path(model, type, batch_size.size, id), questions_path)

                    # we do not compare data used for training across models
                    if type in rating_data:
                        rating_sets[type][model] = model_ratings

                    _get_figure_for_dataset(diagrams, id, model, type).plot(batch_size.size, *_calculate_kappa_for_model(model_ratings))
            
            # calculate kappa of score_1 vs score_1
            for type, selected_model_ratings in rating_sets.items():
                assert len(selected_model_ratings) == len(list(chain.from_iterable(model_sets))), f"Found inconsistent rating set {type}, {selected_model_ratings.keys()}"
                # figures for 
                for model_set in model_sets:
                    ratings, platform = _filter_model_ratings(selected_model_ratings, model_set)
                    kappa, answer_count = _calculate_kappa_for_models(*ratings)
                    log.debug(f"{platform}_ai_v_man - batch_size: {batch_size.size} kappa: {kappa} samples: {answer_count}")
                    if kappa >= -1 and kappa <= 1:
                        _get_figure_for_dataset(diagrams, id, f"{platform}_ai_v_man_score_1", type).plot(batch_size.size, kappa, answer_count)


    # TODO: beautify ######
    all_ai_answers = _answers_to_rating(get_answers(config.get_samples_path("davinci")))
    all_man_answers = _answers_to_rating(get_answers(config.get_samples_path("experts")))
    ai_rated_man_answers = _answers_to_rating(get_answers(config.get_samples_path("davinci_rating_expert_data")))

    chat_gpt_vs_experts = KappaFigure("Kappa of ChatGPT vs Experts", "chat_gpt_vs_experts")
    chat_gpt_vs_experts.plot("GPT-Dataset score_1 vs score_2", *_calculate_kappa_for_model(all_ai_answers))
    chat_gpt_vs_experts.plot("Expert-Dataset score_1 vs score_2", *_calculate_kappa_for_model(all_man_answers))
    chat_gpt_vs_experts.plot("GPT-Rating-Expert-Dataset  score_1 vs score_2", *_calculate_kappa_for_model(ai_rated_man_answers))
    chat_gpt_vs_experts.plot("GPT-VS-Expert score_1 vs score_1", *_calculate_kappa_for_score_types(ai_rated_man_answers, 1, all_man_answers, 1))
    chat_gpt_vs_experts.plot("GPT-VS-Expert score_1 vs score_2", *_calculate_kappa_for_score_types(ai_rated_man_answers, 1, all_man_answers, 2))
    chat_gpt_vs_experts.plot("GPT-VS-Expert score_2 vs score_1", *_calculate_kappa_for_score_types(ai_rated_man_answers, 2, all_man_answers, 1))
    chat_gpt_vs_experts.plot("GPT-VS-Expert score_2 vs score_2", *_calculate_kappa_for_score_types(ai_rated_man_answers, 2, all_man_answers, 2))
    chat_gpt_vs_experts.save(config, f"The AI-Dataset consists of {len(all_ai_answers)} samples.\n The Expert-Dataset consists of {len(all_man_answers)} samples.", True)
    ########################

    # [(ber|xgb)_(man|ai)][(ai|man)-(rating|training)][A-F]
    # [training_data_source][test_data_source][id]
    accumulated_data = _recursive_default_dict()
    for training_data_source, test_data_source_dict in diagrams.items():
        for test_data_source, batch_id_dict in test_data_source_dict.items():

            # x and y for all variants
            x_y_boxplot_data = _recursive_default_dict()
            for batch_id, diagram in batch_id_dict.items():

                #[50, .., 3200] [kappa_1, ... kappa_n]
                d_x, d_y = diagram.get_values()

                log.debug(f"Adding x: {d_x}, y: {d_y} to boxplot matrix")
                assert len(d_x) == len(d_y), f"Found inconsistent data for diagram: {training_data_source} {test_data_source} {batch_id}"

                for i, x in enumerate(d_x):
                    if x in x_y_boxplot_data:
                        x_y_boxplot_data[x].append(d_y[i])
                    else:
                        x_y_boxplot_data[x] = [d_y[i]]

                diagram.save(config)
            
            title = _get_diagram_title(training_data_source, test_data_source)
            _print_boxplot(f"{training_data_source}_{test_data_source}", x_y_boxplot_data, title)

            if test_data_source in rating_data:
                # [(ber|xgb)_(man|ai)][(ai|man)-(rating|training)] = (50 - 3200; A-F)
                accumulated_data[training_data_source][test_data_source] = x_y_boxplot_data

    # boxplots with two datasets (ai vs man)
    for model_set in model_sets:
        for test_data_source in rating_data:
            assert len(model_set) == 2, f"Inconsistent amount of model in set: {model_set}"
            platform = _get_platform_from_model(model_set[0])

            # We need to know which model belongs to which dataset
            data_set_a = (accumulated_data[model_set[0]][test_data_source], model_set[0])
            data_set_b = (accumulated_data[model_set[1]][test_data_source], model_set[1])

            descriptor = f"{platform}_ai_v_man"
            # it is always ai_vs_man
            title = _get_diagram_title(descriptor, test_data_source)
            _print_dual_dataset_boxplot(f"{descriptor}_{test_data_source}", data_set_a, data_set_b, title)