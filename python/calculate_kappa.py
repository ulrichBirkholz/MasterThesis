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

# TODO: step 2 combine multiple boxplots in one diagram using different colors
class KappaFigure():
    def __init__(self, title:str, fileLabel:str) -> None:
        self.answer_count = -1
        self.all_counts_equal = True

        self.filename = f"kappa_{fileLabel}.pdf"
        self.title = title

        self.kappa_values = []
        self.model_sample_sizes = []
    

    def get_values(self):
        return self.model_sample_sizes, self.kappa_values


    def plot(self, model_sample_size, kappa, answer_count):
        if kappa <= -1 or kappa >= 1:
            log.warning(f"Ignore invalid kappa: {kappa}")
            return

        self.kappa_values.append(kappa)
        self.model_sample_sizes.append(model_sample_size)

        if self.answer_count == -1:
            self.answer_count = answer_count
        elif self.answer_count != answer_count:
            # should not happen the data might be inconsistent
            self.all_counts_equal = False
            log.error(f"Data is potentially inconsistent")

            # find the smallest value
            if self.answer_count > answer_count:
                self.answer_count = answer_count


    def _add_figtext(self,figtext):
        plt.figtext(0.5, 1, figtext, fontsize=12, bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})


    def save(self, config: Configuration, figtext:str=None):
        figure = plt.figure()
        plt.title(self.title)

        plt.xlabel('Amount of samples the model was trained with')
        plt.ylabel('Kappa')

        plt.plot(self.model_sample_sizes, self.kappa_values, 'ro')

        if figtext is not None:
            self._add_figtext(figtext)
        elif self.all_counts_equal is True:
            self._add_figtext(f"Each value represents the agreement of exactly {self.answer_count} ratings")
        else:
            self._add_figtext(f"Each value represents the agreement of at leased {self.answer_count} ratings")

        log.debug(f"Save Diagram: {self.filename}")
        figure.savefig(config.get_path_for_datafile(self.filename), bbox_inches='tight')


def _recursive_default_dict():
    return defaultdict(_recursive_default_dict)


def _get_platform_from_model(model_descriptor: str):
    return model_descriptor.split("_")[0].upper()


def _get_diagram_title(model_descriptor: str, answer_descriptor: str):
    platform = _get_platform_from_model(model_descriptor)
    descriptors = {
        "man-training": " ML Model\nrating manually created answers used for training",
        "ai-training": " ML Model\nrating answers created by AI used for training",
        "man-rating": " ML Model\nrating manually created answers",
        "ai-rating": " ML Model\nrating answers created by AI"
    }

    refined_type = "AI-Refined" if model_descriptor.endswith("ai") else "Expert-Refined"
    title = f"{platform} {refined_type}"

    if model_descriptor.endswith("ai_v_man"):
        title = f"{platform} AI-Refined compared with Expert-Refined ML Model\nrating "
    elif answer_descriptor in descriptors:
        title += descriptors[answer_descriptor]
    else:
        log.warning(f"The answer_descriptor {answer_descriptor} is invalid")
        return
    
    return title


def _get_figure_for_dataset(diagrams:Dict, id:str, model_descriptor:str, answer_descriptor:str) -> KappaFigure:
    fileLabel = f"{model_descriptor}_model_{answer_descriptor}_answers_{id}"

    if model_descriptor not in diagrams or answer_descriptor not in diagrams[model_descriptor] or id not in diagrams[model_descriptor][answer_descriptor]:
        diagram_title = _get_diagram_title(model_descriptor, answer_descriptor)
        
        if diagram_title is None:
            return
        
        diagrams[model_descriptor][answer_descriptor][id] = KappaFigure(diagram_title, fileLabel)
    
    return diagrams[model_descriptor][answer_descriptor][id]


# Score 1 is the rating, generated by our trained model
def _get_scores_1(ratings: List[Rating]) -> List[int]:
    return [rating.answer.score_1 for rating in ratings]


# Score 2 is the rating, generated by ChatGPT
def _get_scores_2(ratings: List[Rating]) -> List[int]:
    return [rating.answer.score_2 for rating in ratings]


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

    scores_a = _get_scores_1(ratings_a)
    scores_b = _get_scores_1(ratings_b)

    return cohen_kappa_score(scores_a, scores_b, weights='quadratic'), len(scores_a)


def _calculate_kappa_for_model(ratings:List[Rating]):
    if len(ratings) < 100:
        log.warning(f"Not enough scores: {len(ratings)} to calculate kappa")
        return -5, len(ratings)

    scores_a = _get_scores_1(ratings)
    scores_b = _get_scores_2(ratings)

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

def _print_boxplot(identifier: str, x_y_boxplot_data):
    figure, ax = plt.subplots()

    # sort x ascending
    x_y_boxplot_data_sorted = dict(sorted(x_y_boxplot_data.items()))

    for x, y in x_y_boxplot_data_sorted.items():
        log.debug(f"plotting box for sorted x:{x}; y:{y}")

    # sort each individual y value ascending
    ax.boxplot([sorted(y) for y in x_y_boxplot_data_sorted.values()])
    ax.set_xticklabels(list(x_y_boxplot_data_sorted.keys()))

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
            ratings = _recursive_default_dict()

            for model in chain.from_iterable(model_sets):
                for type in training_data + rating_data:
                    rating_path = config.get_rated_answers_path(model, type, batch_size.size, id)
                    if not os.path.exists(rating_path):
                        log.info(f"The file {rating_path} does not exist")
                        continue

                    model_ratings = get_ratings(config.get_rated_answers_path(model, type, batch_size.size, id), questions_path)

                    # we do not compare data used for training across models
                    if type in rating_data:
                        ratings[type][model] = model_ratings

                    _get_figure_for_dataset(diagrams, id, model, type).plot(batch_size.size, *_calculate_kappa_for_model(model_ratings))
            
            for type, selected_model_ratings in ratings.items():
                assert len(selected_model_ratings) == len(list(chain.from_iterable(model_sets))), f"Found inconsistent rating set {type}, {selected_model_ratings.keys()}"
                for model_set in model_sets:
                    ratings, platform = _filter_model_ratings(selected_model_ratings, model_set)
                    kappa, answer_count = _calculate_kappa_for_models(*ratings)
                    log.debug(f"{platform}_ai_v_man - batch_size: {batch_size.size} kappa: {kappa} samples: {answer_count}")
                    if kappa >= -1 and kappa <= 1:
                        _get_figure_for_dataset(diagrams, id, f"{platform}_ai_v_man", type).plot(batch_size.size, kappa, answer_count)


    # TODO: beautify ######
    all_ai_answers = _answers_to_rating(get_answers(config.get_ai_answers_path()))
    all_man_answers = _answers_to_rating(get_answers(config.get_man_answers_path()))

    score_1_v_2 = KappaFigure("Kappa of score_1 vs. score_2", "score_1_vs_score_2")
    plt.xlabel('Source')
    score_1_v_2.plot("AI-Dataset", *_calculate_kappa_for_model(all_ai_answers))
    score_1_v_2.plot("Expert-Dataset", *_calculate_kappa_for_model(all_man_answers))

    score_1_v_2.save(config, f"The AI-Dataset consists of {len(all_ai_answers)} samples.\n The Expert-Dataset consists of {len(all_man_answers)} samples.")
    ########################

    #
    boxplots = _recursive_default_dict()

    # [model_descriptor][answer_descriptor][id]
    for model_descriptor, answer_descriptor_dict in diagrams.items():
        for answer_descriptor, batch_id_dict in answer_descriptor_dict.items():

            # x and y for all variants
            x_y_boxplot_data = _recursive_default_dict()
            for batch_id, diagram in batch_id_dict.items():

                d_x, d_y = diagram.get_values()

                log.debug(f"Adding x: {d_x}, y: {d_y} to boxplot matrix")
                assert len(d_x) == len(d_y), f"Found inconsistent data for diagram: {model_descriptor} {answer_descriptor} {batch_id}"

                for i, x in enumerate(d_x):
                    if x in x_y_boxplot_data:
                        x_y_boxplot_data[x].append(d_y[i])
                    else:
                        x_y_boxplot_data[x] = [d_y[i]]

                diagram.save(config)
            
            _print_boxplot(f"{model_descriptor}_{answer_descriptor}", x_y_boxplot_data)