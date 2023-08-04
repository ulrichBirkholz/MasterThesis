import xgboost as xgb
from typing import List
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

def _load_components(path, num_class=0):

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

def _save_model_and_vectorizer(model, vectorizer, path):
    os.makedirs(path, exist_ok=True)
    
    model.save_model(f"{path}model")
    with open(f"{path}vectorizer", 'wb') as file:
            pickle.dump(vectorizer, file)
    

def train_model(sample:AnswersForQuestion, path, score_type):

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
    
    # We split the data the same way as for bert, to keep it comparable
    answers_train, answers_test, ratings_train, ratings_test = train_test_split(answers, ratings, test_size=0.2, random_state=42)
    model.fit(vectorizer.fit_transform(answers_train), ratings_train)
    
    predictions = model.predict(vectorizer.transform(answers_test))
    accuracy = accuracy_score(predictions, ratings_test)

    log.info("Accuracy of xg_boost model: %.2f%%" % (accuracy * 100.0))
    
    _save_model_and_vectorizer(model, vectorizer, path)


def rate_answers(path, answers_for_question:AnswersForQuestion, score_type) -> List[Answer]:
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