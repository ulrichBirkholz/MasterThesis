import os
import shutil
import pickle
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import logging as log
from dataclasses import dataclass
from typing import List
from tsv_utils import Answer
import torch

@dataclass
class AnswersForQuestion:
    question_id: str
    question: str
    answers: List[Answer]

# The BERT model accepts input sequences with a maximum length of 512 tokens.
# Limit length to 512 token, ca 230 Words of Lorem Ipsum (represents question + answer + special token ([CLS], [SEP]))
MAX_TOKEN_LENGTH = 512

# currently not wanted
os.environ["WANDB_DISABLED"] = "true"

def _save_dataset(train_dataset, validation_dataset, path):
    os.makedirs(path, exist_ok=True)

    # this is only save to evaluate it on demand
    with open(f"{path}train_dataset.pkl", 'wb') as file:
        pickle.dump(train_dataset, file)
    with open(f"{path}validation_dataset.pkl", 'wb') as file:
        pickle.dump(validation_dataset, file)

def _load_components(path):
    model_and_tokenizer_source = path

    if not os.path.exists(path):
        model_and_tokenizer_source = 'bert-base-uncased'

    model = BertForSequenceClassification.from_pretrained(model_and_tokenizer_source,  num_labels=4)
    tokenizer = BertTokenizer.from_pretrained(model_and_tokenizer_source)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    return model, tokenizer, device


def train_model(sample:AnswersForQuestion, path, epochs):

    # cleanup
    if os.path.exists(path):
        shutil.rmtree(path)

    model, tokenizer, _ = _load_components(path)

    dataset = []
    for answer in sample.answers:
        encodings = tokenizer(sample.question, answer.answer, truncation=True, padding='max_length', max_length=MAX_TOKEN_LENGTH, return_tensors='pt')

        label = int(answer.score_2)
        assert label >= 0 and label <= 4, f"Invalid label {int(answer.score_2)} was detected"
        
        dataset.append({'input_ids': encodings['input_ids'].squeeze(), 'attention_mask': encodings['attention_mask'].squeeze(), 'labels': label})
    
    # Split data into train and validation
    # TODO: we start with 0.2 and evaluate how this affect the rating quality
    #   maybe we adjust this value dynamically depending on the amount of samples
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # Save data before training, the model will be saved by the trainer (see output_dir)
    _save_dataset(train_dataset, validation_dataset, path)

    trainer.train()
    trainer.save_model(path)


def rate_answers(path, answers_for_question:AnswersForQuestion) -> List[Answer]:
    if not os.path.exists(path):
        log.warning(f"The path: {path} does not point to a trained model")
        return [], []

    model, tokenizer, device = _load_components(path)

    predictions = []
    true_values = []
    for answer in answers_for_question.answers:
        encodings = tokenizer(answers_for_question.question, answer.answer, truncation=True, padding='max_length', max_length=MAX_TOKEN_LENGTH, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        answer.score_1 = torch.argmax(outputs.logits, dim=1)
        predictions.append(int(answer.score_1))
        true_values.append(int(answer.score_2))

    cm = confusion_matrix(true_values, predictions)
    return answers_for_question.answers, cm.tolist()