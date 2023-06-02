from typing import List
from json.decoder import JSONDecodeError
import openai
import time
from openai.error import OpenAIError, RateLimitError
import re
import json
import argparse
import logging as log
import re
from tsv_utils import Answer, Question
import hashlib


def generate_answers(api_key, quantity_of_answers, ignore_text_syntax, question) -> List[Answer]:
    openai.api_key = api_key

    # We target 1k Answers per Question
    sampleCategories = [
        "The answers should describe all aspects according to the sample solution",
        "The answers should contain one wrong aspect",
        "The answers should miss at leased one aspect",
        "The answers should be contradictory",
        "The answers should be non domain",
        "The answers should be irrelevant",
        "The answers should be entirely wrong"
    ]

    answers = []
    for category in sampleCategories:
        answers = answers + \
            _generate_answers(
                question, ignore_text_syntax, category, quantity_of_answers)

    return _rate_answers(question, answers, quantity_of_answers)

# TODO: 2 step process? (step one create answers, step two rate them, one request per answer, return rating)
# TODO: update answer.tsv after each request

def _execute_api_call(prompt, max_tokens, temperature, frequency_penalty, presence_penalty):
    model_engine = "text-davinci-003"
    retries = 0
    while True:
        try:
            return openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=1,
                stop=None,
            )
        except RateLimitError:
            #TODO: Adjust this values as needed
            retries += 1
            sleep_duration = 10
            log.warning(f"Rate limit hit. Sleeping for {sleep_duration} before starting retry number: {retries}")
            time.sleep(sleep_duration)
            if retries > 3600:
                log.error(f"Aborting retries")
                break
        except OpenAIError as e:
            log.error(f"OpenAI API caused an error: {e}")
            raise e

def _generate_answers(question:Question, ignore_text_syntax, task, quantity_of_answers=20) -> List[str]:
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

    # Generate answers
    generated_answers = _execute_api_call(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    # Extract answers from OpenAI API response
    answers = []
    for choice in generated_answers.choices:
        log.debug(f"generated answer: {choice.text}")

        # remove new lines
        answer = re.sub(r'\n', ' ', choice.text.strip())
        # find json content
        contains_json = re.search(r'\[(.*?)\]', answer)

        if contains_json:
            answer_str = contains_json.group(0)
            try:
                json_answer = json.loads(answer_str)
                rectified_keys = map(_rectify_keys, json_answer)
                valid_answers = filter(_rectify_answer, rectified_keys)
                answers.extend(valid_answers)
            except JSONDecodeError as e:
                log.error(
                    f"Unable to parse JOSN response: {e}; JSON: {answer_str}")

    # Return list of answers
    return answers

# high temperature can lead to problems creating the JSON
def _rectify_keys(answer):
    return {key.lower(): value for key, value in answer.items()}

def _rectify_answer(answer):
    # if this is incorrect, we need to drop the answer
    if 'answer' not in answer:
        log.error(f"Drop invalid answer: {answer}")
        return False

    # fix minor mistakes
    if 'rating' not in answer:
        formatted_answer, rating = _parse_rating_from_answer(answer['answer'])
        answer['answer'] = formatted_answer
        answer['rating'] = rating
    else:
        answer['rating'] = _validate_rating(answer['rating'])

    return True


def _validate_rating(rating):
    try:
        rating = int(rating)
        if rating >= 0 and rating <= 4:
            return rating
    except:
        log.error(f"Identified invalid rating: {rating}")
    return -1


def _parse_rating_from_answer(answer):
    # It has been observed, that sometimes the rating was added to the answer after a separation of ', ' or ', rating: ' ect.
    patterns = [r', \d$', r'[,]*\s+[Rr]ating:\s*\d$']

    final_answer = answer
    number = -1
    for pattern in patterns:
        match = re.search(pattern, answer)

        if match:
            final_answer = re.sub(pattern, '', answer).strip()
            # get number only
            parsed_number = int(re.search(r'\d$', answer).group())
            if parsed_number >= 0 and parsed_number <= 4:
                number = parsed_number
            break

    return final_answer, number


def _rate_answers(question:Question, answers, ignore_text_syntax) -> List[str]:
    numerated_rated_answers = {
        f"{idx+1}": {"answer": answer['answer'], "rating": answer['rating']} for idx, answer in enumerate(answers)}
    numerated_answers = {f"{idx}": answer['answer']
                         for idx, answer in numerated_rated_answers.items()}
    # Define prompt

    # TODO: category to configuration
    # TODO: datensatzspezische modelle (Powergrading, SRA, CREE usw.)
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

    # Generate answers
    generated_answers = _execute_api_call(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    # Extract answers from OpenAI API response
    rated_answers = []
    for choice in generated_answers.choices:
        log.debug(f"generated rating: {choice.text}")
        # remove new lines
        answer = re.sub(r'\n', ' ', choice.text.strip())
        # find json content
        contains_json = re.search(r'^\{.*\}$', answer)

        if contains_json:
            answer_str = contains_json.group(0).replace("'", '"')
            try:
                json_answer = json.loads(answer_str)
                for answer_id, rating_id in json_answer.items():
                    original_answer = numerated_rated_answers[answer_id]
                    answer_text = original_answer['answer']
                    rated_answers.append(
                        Answer(question.question_id, answer_text, hashlib.md5(answer_text.encode()).hexdigest(), original_answer['rating'], _validate_rating(rating_id)))
            except JSONDecodeError as e:
                log.error(
                    f"Unable to parse JOSN response: {e}; JSON: {answer_str}")

    # Return list of answers
    return rated_answers
