from typing import List
from json.decoder import JSONDecodeError
import openai
import re
import json
import argparse
import logging as log
import re

model_engine = "text-davinci-003"


def generate_answers(api_key, quantity_of_answers, ignore_text_syntax, question, solution, question_id):
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
            __generate_answers(
                question, solution, ignore_text_syntax, category, quantity_of_answers)

    return __rate_answers(question, answers, quantity_of_answers)

# TODO: 2 step process? (step one create answers, step two rate them, one request per answer, return rating)
# TODO: update answer.tsv after each request
# TODO: higher temperature for creating answers, lower for rating them
# The tasks affects the rating


def __generate_answers(question, solution, ignore_text_syntax, task, quantity_of_answers=20) -> List[str]:
    # Define prompt
    basePrompt = f"""Create possible answers for a test.
    The question '{question}' should be answered on {quantity_of_answers} different ways using up to 2 Sentences each.
    Each answer should be rated from 1 to 5 regarding its correctness.
    Present the answers and their ratings in an JSON array of objects formatted like [{{"answer":"answer1", "rating":"7"}}]"""

    if solution is not None:
        basePrompt += f"\nConsider '{solution}' as sample solution containing all relevant aspects."

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
    generated_answers = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=1,
        stop=None,
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
                rectified_keys = map(__rectify_keys, json_answer)
                valid_answers = filter(__rectify_answer, rectified_keys)
                answers.extend(valid_answers)
            except JSONDecodeError as e:
                log.error(
                    f"Unable to parse JOSN response: {e}; JSON: {answer_str}")

    # Return list of answers
    return answers

# high temperature can lead to problems creating the JSON


def __rectify_keys(answer):
    return {key.lower(): value for key, value in answer.items()}


def __rectify_answer(answer):
    # if this is incorrect, we need to drop the answer
    if 'answer' not in answer:
        log.error(f"Drop invalid answer: {answer}")
        return False

    # fix minor mistakes
    if 'rating' not in answer:
        formatted_answer, rating = __parse_rating_from_answer(answer['answer'])
        answer['answer'] = formatted_answer
        answer['rating'] = rating
    else:
        answer['rating'] = __validate_rating(answer['rating'])

    return True


def __validate_rating(rating):
    try:
        rating = int(rating)
        if rating >= 1 and rating <= 5:
            return rating
    except:
        log.error(f"Identified invalid rating: {rating}")
    return -1


def __parse_rating_from_answer(answer):
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
            if parsed_number >= 1 and parsed_number <= 5:
                number = parsed_number
            break

    return final_answer, number


def __rate_answers(question, answers, ignore_text_syntax) -> List[str]:
    numerated_rated_answers = {
        f"{idx+1}": {"answer": answer['answer'], "rating": answer['rating']} for idx, answer in enumerate(answers)}
    numerated_answers = {f"{idx}": answer['answer']
                         for idx, answer in numerated_rated_answers.items()}
    # Define prompt
    prompt = f"""I am an AI trained to score responses based on a five-point scale of relevance, coherence, and completeness.
    The answers to be rated are formatted as JSON in the following matter {{"answer_id1":"answer1"}} 
    Please evaluate the following answers based on these criteria:

    Question: {question}
    Answers: {numerated_answers}

    rating_id: criteria
    1: Incorrect or irrelevant
    2: Partially correct
    3: Somewhat correct
    4: Mostly correct
    5: Completely correct

    Present the ratings in a JSON format like {{"answer_id1":"rating_id1"}}"""

    if ignore_text_syntax:
        prompt += "\nIgnore spelling or punctuation mistakes for the evaluation."

    # print(f"Prompt: {prompt}")

    # Set up parameters for generating answers
    max_tokens = 3036
    temperature = 0.4
    frequency_penalty = 0.6
    presence_penalty = 0.2

    # Generate answers
    generated_answers = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=1,
        stop=None,
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
                    answer = numerated_rated_answers[answer_id]
                    rated_answers.append(
                        {'answer': answer['answer'], 'rating1': answer['rating'], 'rating2': __validate_rating(rating_id)})
            except JSONDecodeError as e:
                log.error(
                    f"Unable to parse JOSN response: {e}; JSON: {answer_str}")

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
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    parser.add_argument('--ignore_text_syntax', action='store_true',
                        help='Ignore spelling or punctuation mistakes for the evaluation')
    args = parser.parse_args()

    # Print list of answers, the question_id is always '1' hence there is only one question
    for i, answer in enumerate(generate_answers(args.api_key, args.quantity, args.ignore_text_syntax, args.question, args.answer, 1)):
        print(f"{i+1}. {answer}")
