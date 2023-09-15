import xgboost as xgb
from typing import List, Tuple, Any
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from bert_utils import AnswersForQuestion
from sklearn.metrics import confusion_matrix
from tsv_utils import Answer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os
import shutil
import logging as log


def _load_components(path:str, num_class:int=0) -> Tuple[xgb.XGBClassifier, TfidfVectorizer]:
    """ Loads the model and vectorizer.
    Given a specific path, this function attempts to load the XGBClassifier and TfidfVe

    Args:
        path (str): The directory path where the model is expected to be stored
        num_class (int, optional): The number of individual categories, present in the dataset. Defaults to 0.

    Returns:
        Tuple[xgb.XGBClassifier, TfidfVectorizer]: A tuple containing the XGBClassifier model and the TfidfVectorizer in that order.
    """

    if os.path.exists(path):
        model = xgb.XGBClassifier()
        model.load_model(f"{path}model")
        
		# the vectorizer learns from the training data during the fit_transform process
        with open(f"{path}vectorizer", 'rb') as file:
            vectorizer = pickle.load(file)

    else:
        # multiclass classification problem (0 - 3), small datasets may not contain all categories
        model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_class, random_state=42)     
        vectorizer = TfidfVectorizer()
    
    return model, vectorizer


def _save_model_and_vectorizer(model:xgb.XGBClassifier, vectorizer:TfidfVectorizer, path:str) -> None:
    """ Save the model and the vectorizer

    Args:
        model (xgb.XGBClassifier): Model to be saved
        vectorizer (TfidfVectorizer): Vectorizer to be saved
        path (str): Path to the models folder
    """
    os.makedirs(path, exist_ok=True)
    
    model.save_model(f"{path}model")
    with open(f"{path}vectorizer", 'wb') as file:
            pickle.dump(vectorizer, file)
    

def train_model(sample:AnswersForQuestion, path:str, score_type:int) -> None:
    """ Train a XG-Boost model using the given answers for a particular question, and save the trained model to the specified directory.

    This method takes sample answers related to a specific question, trains a XG-Boost model on it for the specified
    number of epochs, and then saves the trained model to the provided directory path

    Args:
        sample (AnswersForQuestion): A data structure containing the question and its corresponding answers
        path (str): The directory path where the trained model will be saved
        epochs (int): Specifies the number of times the learning algorithm will work through the entire training dataset. 
                      The more epochs, the more the model will be refined, but care should be taken to avoid overfitting
        score_type (int): The score type (1 or 2) to be used
    """
    # cleanup
    if os.path.exists(path):
        shutil.rmtree(path)
    
    answers = []
    ratings = []
    for answer in sample.answers:
        rating = int(getattr(answer, f'score_{score_type}'))
        assert rating >= 0 and rating < 4, f"Invalid rating {rating} was detected"
        answers.append(answer.answer)
        ratings.append(rating)
    
    model, vectorizer = _load_components(path, len(set(ratings)))
    
    # We split the data the same way as for BERT, to keep it comparable
    answers_train, answers_test, ratings_train, ratings_test = train_test_split(answers, ratings, test_size=0.2, random_state=42)
    model.fit(vectorizer.fit_transform(answers_train), ratings_train)
    
    predictions = model.predict(vectorizer.transform(answers_test))
    accuracy = accuracy_score(predictions, ratings_test)

    log.info("Accuracy of xg_boost model: %.2f%%" % (accuracy * 100.0))
    
    _save_model_and_vectorizer(model, vectorizer, path)


def test_model(path:str, answers_for_question:AnswersForQuestion, score_type:int) -> Tuple[List[Answer], Any]:
    """ Evaluates a given set of answers using a trained XG-Boost model and returns annotated answers.

    This method loads a trained XG-Boost model from the specified directory path and then rates the provided set of 
    answers. Each answer is annotated with two scores: the predicted score from the XG-Boost model and the true 
    value score from the dataset.

    Args:
        path (str): The directory path where the trained model is saved
        answers_for_question (AnswersForQuestion): A data structure containing a question and its corresponding 
                                                   answers that are to be rated by the model
        score_type (int): Indicator for which score type is considered the true value (1 or 2)

    Returns:
        Tuple[List[Answer], Any]: A tuple containing:
                      1. List[Answer]: Annotated answers. Each answer comes with:
                         - Score 1: The predicted score from the XG-Boost model.
                         - Score 2: The true value score from the dataset, based on the selected score_type.
                      2. Any: The confusion matrix resulting from the test.
    """
    log.debug(f"rating answers with xgb: {path}")

    if not os.path.exists(path):
        log.warning(f"The path: {path} does not point to a trained model")
        return [], []

    model, vectorizer = _load_components(path)

    answers = []
    ratings = []
    for answer in answers_for_question.answers:
        rating = int(answer.score_2)
        assert rating >= 0 and rating < 4, f"Invalid rating {rating} was detected"
        answers.append(answer.answer)
        ratings.append(rating)

    predictions = model.predict(vectorizer.transform(answers))

    for answer, prediction in zip(answers_for_question.answers, predictions):
        original_value = int(getattr(answer, f'score_{score_type}'))
        answer.score_1 = prediction
        answer.score_2 = original_value

    cm = confusion_matrix(ratings, predictions)
    return answers_for_question.answers, cm.tolist()