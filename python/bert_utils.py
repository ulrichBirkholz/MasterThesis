import os
import shutil
import pickle
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import IntervalStrategy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import logging as log
from dataclasses import dataclass
from typing import List, Tuple, Any
from tsv_utils import Answer

@dataclass
class AnswersForQuestion:
    """ Represents a collection of answers associated with a specific question.
    
    This dataclass holds the question ID, the question text, and a list of corresponding answers.

    Attributes:
        question_id (str): Unique identifier for the question.
        question (str): The text of the question.
        answers (List[Answer]): A list of answer objects related to the question.
    """
    question_id: str
    question: str
    answers: List[Answer]

# The BERT model accepts input sequences with a maximum length of 512 tokens.
# Limit length to 512 token, ca 230 Words of Lorem Ipsum (represents question + answer + special token ([CLS], [SEP]))
MAX_TOKEN_LENGTH = 512

# currently not wanted
os.environ["WANDB_DISABLED"] = "true"


def _save_dataset(train_dataset:List[Answer], validation_dataset:List[Answer], path:str) -> None:
    """ Save the samples used to train and validate the model
    This allows later to trace which set of answers produced which result

    Args:
        train_dataset (List[Answer]): Dataset used to train the BERT model
        validation_dataset (List[Answer]): Dataset used to validate the BERT model
        path (str): Path to the models folder
    """
    os.makedirs(path, exist_ok=True)

    # this is only save to evaluate it on demand
    with open(f"{path}train_dataset.pkl", 'wb') as file:
        pickle.dump(train_dataset, file)
    with open(f"{path}validation_dataset.pkl", 'wb') as file:
        pickle.dump(validation_dataset, file)


def _load_components(path:str) -> Tuple[Any, Any, Any]:
    """ Loads the model, tokenizer, and device.
    Given a specific path, this function attempts to load the BertForSequenceClassification, 
    BertTokenizer, and torch device. If the path doesn't correspond to an existing folder, 
    it defaults to using 'bert-base-uncased'

    Args:
        path (str): The directory path where the model is expected to be stored

    Returns:
        Tuple[Any, Any, Any]: A tuple containing the BertForSequenceClassification model, 
        BertTokenizer, and torch device in that order.
    """
    model_and_tokenizer_source = path

    if not os.path.exists(path):
        model_and_tokenizer_source = 'bert-base-uncased'

    model = BertForSequenceClassification.from_pretrained(model_and_tokenizer_source,  num_labels=4)
    tokenizer = BertTokenizer.from_pretrained(model_and_tokenizer_source)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    return model, tokenizer, device


def train_model(sample:AnswersForQuestion, path:str, epochs:int, score_type:int) -> None:
    """ Train a BERT model using the given answers for a particular question, and save the trained model to the specified directory.

    This method takes sample answers related to a specific question, trains a BERT model on it for the specified
    number of epochs, and then saves the trained model to the provided directory path

    Args:
        sample (AnswersForQuestion): A data structure containing the question and its corresponding answers
        path (str): The directory path where the trained model will be saved
        epochs (int): Specifies the number of times the learning algorithm will work through the entire training dataset. 
                      The more epochs, the more the model will be refined, but care should be taken to avoid overfitting
        score_type (int): The score type (either 1 or 2) to be used
    """
    # cleanup
    if os.path.exists(path):
        shutil.rmtree(path)

    model, tokenizer, _ = _load_components(path)

    dataset = []
    for answer in sample.answers:
        encodings = tokenizer(sample.question, answer.answer, truncation=True, padding='max_length', max_length=MAX_TOKEN_LENGTH, return_tensors='pt')

        label = int(getattr(answer, f'score_{score_type}'))
        assert label >= 0 and label < 4, f"Invalid label {label} was detected"
        
        dataset.append({'input_ids': encodings['input_ids'].squeeze(), 'attention_mask': encodings['attention_mask'].squeeze(), 'labels': label})
    
    # Split data into train and validation
    train_dataset, validation_dataset = train_test_split(dataset, test_size=0.2)

    training_args = TrainingArguments(
        output_dir=path,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=1e-5,
        logging_dir=f"{path}logs",
        load_best_model_at_end=True,
        evaluation_strategy=IntervalStrategy.STEPS
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset
    )

    # Save data before training, the model will be saved by the trainer (see output_dir)
    _save_dataset(train_dataset, validation_dataset, path)

    trainer.train()
    trainer.save_model(path)
    tokenizer.save_pretrained(path)


def test_model(path:str, answers_for_question:AnswersForQuestion, score_type:int) -> Tuple[List[Answer], Any]:
    """ Evaluates a given set of answers using a trained BERT model and returns annotated answers.

    This method loads a trained BERT model from the specified directory path and then rates the provided set of 
    answers. Each answer is annotated with two scores: the predicted score from the BERT model and the true 
    value score from the dataset.

    Args:
        path (str): The directory path where the trained model is saved
        answers_for_question (AnswersForQuestion): A data structure containing a question and its corresponding 
                                                   answers that are to be rated by the model
        score_type (int): Indicator for which score type is considered the true value (either 1 or 2)

    Returns:
        Tuple[List[Answer], Any]: A tuple containing:
                      1. List[Answer]: Annotated answers. Each answer comes with:
                         - Score 1: The predicted score from the BERT model.
                         - Score 2: The true value score from the dataset, based on the selected score_type.
                      2. Any: The confusion matrix resulting from the test.
    """
    log.debug(f"rating answers with bert: {path}")

    if not os.path.exists(path):
        log.warning(f"The path: {path} does not point to a trained model")
        return [], []

    model, tokenizer, device = _load_components(path)

    predictions = []
    true_values = []
    results = []
    for answer in answers_for_question.answers:
        encodings = tokenizer(answers_for_question.question, answer.answer, truncation=True, padding='max_length', max_length=MAX_TOKEN_LENGTH, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        result = int(torch.argmax(outputs.logits, dim=1).item())
        original_value = int(getattr(answer, f'score_{score_type}'))

        results.append(Answer(answer.question_id, answer.answer, answer.answer_id, result, original_value))

        predictions.append(result)
        true_values.append(original_value)

    cm = confusion_matrix(true_values, predictions)
    return results, cm.tolist()