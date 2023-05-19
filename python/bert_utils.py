import torch
import os
import shutil
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import logging as log

# The BERT model accepts input sequences with a maximum length of 512 tokens.
# Limit length to 512 token, ca 230 Words of Lorem Ipsum (represents question + answer + special token ([CLS], [SEP]))
MAX_TOKEN_LENGTH = 512


def _save_model_and_tensor(model, all_sample_ids, label, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(f"{path}")
    torch.save(label, f"{path}/label.pt")

    # The input tensors are now a list of 1D tensors, which can't be saved directly.
    # We need to save them separately:
    for i, sample_ids in enumerate(all_sample_ids):
        torch.save(sample_ids, f"{path}/sample_ids_{i}.pt")

def _format_tokens(question_tokens, answer_tokens):
    tokens = ['[CLS]'] + question_tokens + \
        ['[SEP]'] + answer_tokens + ['[SEP]']
    if len(tokens) > MAX_TOKEN_LENGTH:
        log.error(
            f"The sequence '{tokens}' is too long for the model to handle.")
        raise ValueError(
            f"Token length {len(tokens)} exceeds maximum of {MAX_TOKEN_LENGTH}")
    return tokens

def _load_components(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = f"{path}"
    label_path = f"{path}/label.pt"

    if os.path.exists(model_path) and os.path.isfile(os.path.join(model_path, 'pytorch_model.bin')):
        model = _create_model(model_path)
    else:
        model = _create_model('bert-base-uncased')
        log.info("No pretrained model found, using default bert-base-uncased model.")

    file_number = 0
    sample_ids = []
    while os.path.exists(f"{path}/sample_ids_{file_number}.pt"):
        sample_ids.append(torch.load(f"{path}/sample_ids_{file_number}.pt").long())
        file_number += 1

	# one tensor for an array of label
    if os.path.exists(label_path):
        label_tensor = torch.load(label_path)
    else:
        label_tensor = torch.tensor([])
        log.info("No label tensor found, using an empty tensor.")

    return device, model.to(device), sample_ids, label_tensor

def _collate_fn(batch):
    # batch is a list of tensor samples
    # Pad the sequence and return as a single tensor
    return pad_sequence(batch, batch_first=True)

# num_labels = 6: 0 + 1 - 5
def _create_model(model_path, num_labels=6):
    model = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model

# possible modes: 'new', 'continue', 'extend'
# 'new' do not load and use given samples
# 'extend' do load and use samples
# 'continue' do load and ignore samples
def train_model(samples, path, epochs, mode='new'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    all_sample_tensors = []
    labels = []

    # cleanup
    if mode == 'new':
        shutil.rmtree(path)
        os.mkdir(path)

    if mode == 'new' or mode == 'extend':
        for sample in samples:

            question_tokens = tokenizer.tokenize(sample["question"])
            for answer in sample["answers"]:
                answer_tokens = tokenizer.tokenize(answer['answer'])
                sample_tokens = _format_tokens(question_tokens, answer_tokens)
                
                sample_ids = tokenizer.convert_tokens_to_ids(sample_tokens)
                all_sample_tensors.append(torch.tensor(sample_ids))
                labels.append(int(answer['score']))

    # one tensor for an array of label
    label_tensor = torch.tensor(labels)
    device, model, loaded_sample_tensors, loaded_label_tensor = _load_components(path)
    
    labels_tensors = torch.cat((loaded_label_tensor, label_tensor), dim=0)
    
    all_sample_tensors = all_sample_tensors + loaded_sample_tensors
    assert len(all_sample_tensors) == labels_tensors.size(0), f"Assertion failed: The number of sample tensors: {len(all_sample_tensors)} must equal the number of labels: {labels_tensors.size(0)}"
    
    label_tensors_loader = DataLoader(labels_tensors.long(), batch_size=32, drop_last=False)

    # Combine sample ids, create sample tensors and create a DataLoader for the sample tensors
    # Padding tensors, (add zeros until every id is of equal length)
	# this is required for batch processing answers
    sample_tensors_loader = DataLoader(
        all_sample_tensors, batch_size=32, collate_fn=_collate_fn, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # categorization task
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for sample_tensor_batch, labels_tensor_batch in zip(sample_tensors_loader, label_tensors_loader):
            # debug
            log.debug(f'Sample tensor batch size: {sample_tensor_batch.size()}, Labels tensor batch size: {labels_tensor_batch.size()}')
            assert sample_tensor_batch.size(0) == labels_tensor_batch.size(0), f"Assertion failed: The batch size of samples tensors: {sample_tensor_batch.size(0)} and label tensors: {labels_tensor_batch.size(0)} must be equal"
            
            optimizer.zero_grad()
            outputs = model(sample_tensor_batch.to(device))
            loss = criterion(outputs.logits, labels_tensor_batch)
            loss.backward()
            optimizer.step()
        # save after every epoche, so we can continue without much losses
        _save_model_and_tensor(model, all_sample_tensors, labels_tensors, path)

    return model

def rate_answer(path, questions_and_answers):
    _, model, _, _ = _load_components(path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    rated_answers = []
    for question_and_answers in questions_and_answers:
        question_tokens = tokenizer.tokenize(question_and_answers["question"])

        for answer in question_and_answers["answers"]:
            answer_tokens = tokenizer.tokenize(answer["answer"])
            tokens = _format_tokens(question_tokens, answer_tokens)
            
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            tensor_ids = torch.tensor([token_ids])
            tensor_ids = tensor_ids.to(model.device)
            with torch.no_grad():
                outputs = model(tensor_ids)

            print(f"Rate: {outputs}")

            # Average, TODO: check if this is supposed to be tensor containing multiple values
            average_value = torch.mean(outputs.logits).item()

            rated_answers.append([
                question_and_answers["question_id"],
                answer["answer_id"],
                average_value
                                
            ])
    
    return rated_answers
