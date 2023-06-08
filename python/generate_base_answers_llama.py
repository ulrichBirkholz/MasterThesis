from typing import List
from json.decoder import JSONDecodeError
import time
from tsv_utils import Question
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, GenerationConfig
import re
import os
import json
import argparse
import logging as log
import re
from datetime import datetime
import sys


def generate_answers(path, name, quantity_of_answers, ignore_text_syntax, question: Question):
    text_generator = _load_components(path, name)

    # We target 1k Answers per Question
    sampleCategories = [
        "The answers should describe all aspects",
        #"The answers should contain one wrong aspect",
        #"The answers should miss at leased one aspect",
        #"The answers should be contradictory",
        #"The answers should be non domain",
        #"The answers should be irrelevant",
        "The answers should be entirely wrong"
    ]

    answers = []
    for category in sampleCategories:
        answers = answers + \
            _generate_answers(text_generator,
                question, ignore_text_syntax, category, quantity_of_answers)

    return _rate_answers(text_generator, question, answers, quantity_of_answers)

# TODO: 2 step process? (step one create answers, step two rate them, one request per answer, return rating)
# TODO: update answer.tsv after each request

def _execute_prompt(text_generator, prompt, max_new_tokens, temperature):
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] - Execute Prompt: {prompt}")
    try:
        # Generate answers
        output = text_generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    except BaseException as e:
        end_time = datetime.now()
        print(f"[{end_time.strftime('%H:%M:%S')}] - Aborted execution")
        print(f"Duration: {(end_time - start_time).total_seconds()}")
        sys.exit("Exception occurred: {e}")

    end_time = datetime.now()
    print(f"[{end_time.strftime('%H:%M:%S')}] - Produced Output: {output}")
    print(f"Generated Answers: {output[0]['generated_text']}")
    print(f"Duration: {(end_time - start_time).total_seconds()}")

    return output[0]['generated_text']
     

def _load_components(path, name):
	# Check if the model has been saved locally
	if os.path.exists(path):
		log.info(f"Loading model from {path}")
		tokenizer = LlamaTokenizer.from_pretrained(path)
		model = LlamaForCausalLM.from_pretrained(path)
	else:
		log.info("Downloading and loading model {name}")
		tokenizer = LlamaTokenizer.from_pretrained(name)
		model = LlamaForCausalLM.from_pretrained(name)
		
		log.debug("Saving model to", path)
		os.makedirs(f"{path}", exist_ok=True)
		tokenizer.save_pretrained(path)
		model.save_pretrained(path)

	return pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1)  # use device=0 for GPU

def _generate_answers(text_generator, question:Question, ignore_text_syntax, task, quantity_of_answers=20) -> List[str]:
    # Define prompt
    basePrompt = f"""Create possible answers for a test.
    The question '{question.question}' should be answered on {quantity_of_answers} different ways using up to 2 Sentences each.
    Each answer should be rated from 0 to 4 regarding its correctness.
    Present the answers and their ratings in an JSON array of objects formatted like [{{"answer":"answer1", "rating":"7"}}]"""

    if question.sample_answer is not None:
        basePrompt += f"\nConsider '{question.sample_answer}' as sample solution containing all relevant aspects."

    if ignore_text_syntax:
        basePrompt += "\nIgnore spelling or punctuation mistakes for the evaluation."

    # prompt = basePrompt
    prompt = basePrompt + f"\n{task}"

    # Set up parameters for generating answers
    max_tokens = 3036
    temperature = 0.8
    frequency_penalty = 0.2
    presence_penalty = 0.6

    _execute_prompt(text_generator,
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature)

    # Extract answers from OpenAI API response
    answers = []
    # TODO output to array

    # Return list of answers
    return answers

def _validate_rating(rating):
    try:
        rating = int(rating)
        if rating >= 0 and rating <= 4:
            return rating
    except:
        log.error(f"Identified invalid rating: {rating}")
    return -1


def _rate_answers(text_generator, question:Question, answers, ignore_text_syntax) -> List[str]:
    numerated_rated_answers = {
        f"{idx+1}": {"answer": answer['answer'], "rating": answer['rating']} for idx, answer in enumerate(answers)}
    numerated_answers = {f"{idx}": answer['answer']
                         for idx, answer in numerated_rated_answers.items()}
    # Define prompt

    # TODO: category to configuration
    # TODO: datensatzspezische modelle
    # TODO: kappa man vs machine
    # ChatGPT should also rate manually annotated test answers
    # Trained models should rate equal answers
    prompt = f"""I am an AI trained to score responses based on a five-point scale of relevance, coherence, and completeness.
    The answers to be rated are formatted as JSON in the following matter {{"answer_id1":"answer1"}} 
    Please evaluate the following answers based on these criteria:

    Question: {question.question}
    Answers: {numerated_answers}

    rating_id: criteria
    0: Incorrect or irrelevant
    1: Partially correct
    2: Somewhat correct
    3: Mostly correct
    4: Completely correct

    Present the ratings in a JSON format like {{"answer_id1":"rating_id1"}}"""

    if ignore_text_syntax:
        prompt += "\nIgnore spelling or punctuation mistakes for the evaluation."

    # print(f"Prompt: {prompt}")

    # Set up parameters for generating answers
    max_tokens = 2000
    temperature = 0.4
    frequency_penalty = 0.6
    presence_penalty = 0.2

    _execute_prompt(text_generator,
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature)

    # Extract answers from OpenAI API response
    rated_answers = []

    # Return list of answers
    return rated_answers


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", help="The question for which to generate answers")
    parser.add_argument('--answer', default=None,
                        help='A sample answer (optional)')
    parser.add_argument('quantity', default=20,
                        help='Amount of created Answers')
    parser.add_argument('--ignore_text_syntax', action='store_true',
                        help='Ignore spelling or punctuation mistakes for the evaluation')
    args = parser.parse_args()

    # Print list of answers, the question_id is always '1' hence there is only one question
    for i, answer in enumerate(generate_answers(args.quantity, args.ignore_text_syntax, args.question, args.answer, 1)):
        print(f"{i+1}. {answer}")
