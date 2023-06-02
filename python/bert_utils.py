import os
import shutil
import pickle
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import logging as log
from dataclasses import dataclass
from typing import List
from tsv_utils import Answer

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

def _format_dataset(dataset):
    for i in range(len(dataset)):
        # tokenized input
        dataset[i]['input_ids'] = torch.tensor(dataset[i]['input_ids'])
        # identify which tokens are words and which are padding
        dataset[i]['attention_mask'] = torch.tensor(dataset[i]['attention_mask'])
        # stcore of intput
        dataset[i]['labels'] = torch.tensor(dataset[i]['labels'])

def _save_model_and_dataset(tokenizer, trainer, train_dataset, validation_dataset, path):
    os.makedirs(f"{path}", exist_ok=True)
    trainer.save_model(f"{path}")
    tokenizer.save_pretrained(f"{path}")

    with open(f"{path}train_dataset.pkl", 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(f"{path}validation_dataset.pkl", 'wb') as f:
        pickle.dump(validation_dataset, f)

def _load_components(path):
    model_and_tokenizer_source = path

    if not os.path.exists(path):
        model_and_tokenizer_source = 'bert-base-uncased'

    model = BertForSequenceClassification.from_pretrained(model_and_tokenizer_source,  num_labels=5)
    tokenizer = BertTokenizer.from_pretrained(model_and_tokenizer_source)

    train_dataset = []
    validation_dataset = []
    if os.path.exists(path):
        with open(f"{path}train_dataset.pkl", 'rb') as f:
            train_dataset = pickle.load(f)
        with open(f"{path}validation_dataset.pkl", 'rb') as f:
            validation_dataset = pickle.load(f)

    return train_dataset, validation_dataset, model, tokenizer

# possible modes: 'new', 'continue', 'extend'
# 'new' do not load and use given samples
# 'extend' do load and use samples
# 'continue' do load and ignore samples
def train_model(samples:List[AnswersForQuestion], path, epochs, mode='new'):

    # cleanup
    if mode == 'new' and os.path.exists(path):
        shutil.rmtree(path)

    loaded_train_dataset, loaded_validation_dataset, model, tokenizer = _load_components(path)

    dataset = []
    if mode == 'new' or mode == 'extend':
        for sample in samples:
            for answer in sample.answers:
                encodings = tokenizer(sample.question, answer.answer, truncation=True, padding='max_length', max_length=MAX_TOKEN_LENGTH)

                label = int(answer.score_2)
                assert label >= 0 and label <= 4, f"Invalid label {int(answer.score_2)} was detected"
                dataset.append({'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask'], 'labels': label})
    
    # Split data into train and validation
    # TODO: we start with 0.2 and evaluate how this affect the rating quality
    train_dataset, validation_dataset = train_test_split(dataset, test_size=0.2)

    _format_dataset(train_dataset)
    _format_dataset(validation_dataset)

    combined_train_dataset = train_dataset + loaded_train_dataset
    combined_validation_dataset = validation_dataset + loaded_validation_dataset

    training_args = TrainingArguments(
        output_dir=path,                 # output directory
        #save_total_limit=2,
        num_train_epochs=epochs,         # total number of training epochs also try 3
        per_device_train_batch_size=16,  # batch size per device during training # TODO: depends on GPU memory, worth a parameter?
        per_device_eval_batch_size=64,   # batch size for evaluation # TODO: depends on GPU memory, worth a parameter?
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        learning_rate=1e-5,              # learning rate
        logging_dir=f"{path}logs",       # directory for storing logs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_train_dataset,
        eval_dataset=combined_validation_dataset,
    )

    # Save data before training, the model will be updated by the trainer (see output_dir)
    _save_model_and_dataset(tokenizer, trainer, combined_train_dataset, combined_validation_dataset, path)

    trainer.train()

    #_save_model_and_dataset(tokenizer, trainer, combined_train_dataset, combined_validation_dataset, path)

def rate_answer(path, answers_for_questions:List[AnswersForQuestion]) -> List[Answer]:
    _, _, model, tokenizer = _load_components(path)

    rated_answers = []
    for answers_for_question in answers_for_questions:
        for answer in answers_for_question.answers:
            # Tokenize and format the question-answer pair
            encodings = tokenizer(answers_for_question.question, answer.answer, truncation=True, padding='max_length', max_length=MAX_TOKEN_LENGTH)
            input_ids = torch.tensor(encodings['input_ids']).unsqueeze(0)  # add batch dimension
            attention_mask = torch.tensor(encodings['attention_mask']).unsqueeze(0)  # add batch dimension

            # Make prediction
            with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
                outputs = model(input_ids, attention_mask)
                logits = outputs.logits

            # Compute predicted rating
            answer.score_1 = torch.argmax(logits, dim=1).item()
            rated_answers.append(answer)

    return rated_answers
