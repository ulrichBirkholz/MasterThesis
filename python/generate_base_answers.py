from typing import List, Iterator, Generator
from json.decoder import JSONDecodeError
import openai
import time
from openai.error import OpenAIError, RateLimitError
import re
import json
#import config
import logging as log
import re
from tsv_utils import Answer, Question, KeyElement
import hashlib
import tiktoken
import random
from dataclasses import dataclass

@dataclass
class SampleCategory:
    number_of_key_elements: int
    task: str

def _count_token(prompt:str) -> int:
    # TODO: model to config
    #enc = tiktoken.encoding_for_model("gpt-4")
    enc = tiktoken.encoding_for_model("text-davinci-003")

    return len(enc.encode(prompt))

def generate_answers(api_key, question:Question, key_elements:List[KeyElement]) -> Generator[List[Answer], None, None]:
    openai.api_key = api_key

    for prompt in generate_answer_prompt(question, key_elements):
        retries = 0
        # TODO: configure amount of retries
        max_retries = 5
        while retries < max_retries:
            try:
                yield _generate_answers(
                        question, prompt)
                break
            except JSONDecodeError as e:
                retries += 1
                log.error(
                    f"Unable to parse JSON response: {e}")
            except Exception as e:
                retries += 1
                log.error(
                    f"Unable to generate Answers: {e}")

        if retries >= max_retries:
            # TODO: add more info like question?
            log.error(f"Exceeded {max_retries} retries, the creation of answers will be aborted")

def _execute_api_call(prompt, max_tokens, temperature, frequency_penalty, presence_penalty):
    model_engine = "text-davinci-003"
    retries = 0
    max_retries = 3600
    while retries < max_retries:
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
        except OpenAIError as e:
            log.error(f"OpenAI API caused an error: {e}")
            retries += 1
            sleep_duration = 10
            log.warning(f"Rate limit hit. Sleeping for {sleep_duration} before starting retry number: {retries}")
            time.sleep(sleep_duration)
            #raise e

    if retries >= max_retries:
        log.error(f"To many retries")

def _add_key_elements(prompt:str, key_elements:List[KeyElement]) -> str:
    if len(key_elements) > 0:
        result = prompt + "\n\nThe key elements for this task are:"
        for element in key_elements:
            result += f"\n- {element.element}"
    return result + "\n"

def _setup_category(category:SampleCategory, key_elements:List[KeyElement]):
    random_elements = random.sample(key_elements, category.number_of_key_elements)
    if len(key_elements) > 0:
        line = f"{category.task}:"
        for element in random_elements:
            line += f"\n- {element.element}"
    return line
    

# 10 x 21 x 20 = 4200 individual prompts
# The texts have been enhanced with the assistance of Chat-GPT, optimizing their comprehensibility for Davinci-003.
# The elements use a simple phrasing to make them easily understandable
def generate_answer_prompt(question:Question, key_elements:List[KeyElement]):

    # 10 categories of correctness
    sample_categories = [
        SampleCategory(len(key_elements), "avoid any of the following information in the response"),
        SampleCategory(len(key_elements), "provide an analogy or metaphor for comprehension but avoid any of the following information in the response"),
        SampleCategory(1, "include a paraphrase of all of the following information in the response"),
        SampleCategory(2, "include a paraphrase of all of the following information in the response"),
        SampleCategory(3, "include a paraphrase of all of the following information in the response"),
        SampleCategory(3, "include a paraphrase of all of the following information in an unrelated or incorrect context in the response"),
        SampleCategory(random.randint(3, len(key_elements)), "include a paraphrase of all of the following information in the response"),
        SampleCategory(random.randint(1, len(key_elements)), "include a paraphrase of all of the following information but deviate from the main topic in the response"),
        SampleCategory(len(key_elements), "create a response that contradicts the following information, with reasoning"),
        SampleCategory(len(key_elements), "provide an analogy or metaphor for comprehension but avoid any of the following information in the response")
    ]

    # 21 combinations of tone syntax, mood, voice and style
    # This could be broken down in tones ["Neutral", ...] 
    #   and syntaxes ["complex scientific jargon", ...] 
    #   to further increase the number of variations of styles
    styles = [
        "Neutral tone with complex scientific jargon",
        "Academic tone with simple, direct language",
        "Didactic tone with compound sentences",
        "Curious tone with question-based syntax",
        "Energetic tone with assertive language",
        "Respectful tone with active syntax",
        "Critical tone with passive syntax",
        "Honest tone with concise syntax",
        "Enthusiastic tone with cause-and-effect syntax",
        "Pragmatic tone with precise syntax",
        "Serene tone with layered syntax",
        "Serious tone with procedural syntax",
        "Imaginative tone with parallel syntax",
        "Contemplative tone with contrasting syntax",
        "Clear tone with concise syntax",
        "Realistic tone with technical syntax",
        "Optimistic tone with balanced syntax",
        "Concise tone with analytical syntax",
        "Engaging tone with narrative syntax",
        "Curious tone with question-based syntax"
        "Brief, less than 8 words"
    ]

    # 20 roles the AI embodies by creating the answers
    # It would be possible to provide a list of distinct idioms if this is not working properly
    roles = [
        # Beginner
        "an American Southerner using regional vernacular, new to the subject",
        "an ESL student with basic English skills, discovering the subject",
        "a Caribbean native with beginner English skills, learning the subject with Creole influences",
        "an English-speaking student learning the topic for the first time",
        "a non-native English speaker with limited English, navigating basic concepts of the topic",
        
        # Intermediate
        "an American Midwesterner with intermediate subject knowledge, using regional dialect",
        "a Scot with foundational subject knowledge, integrating Scottish slang",
        "an East African English speaker studying the topic at an intermediate level with Swahili influences",
        "an undergraduate student with a basic grasp on the subject",
        "a non-specialist approaching the subject from a different field's perspective",

        # Advanced
        "a native English speaking senior researcher with considerable subject knowledge",
        "a Bostonian with substantial understanding of the subject, using local dialect",
        "a South African with significant subject understanding, integrating local colloquialisms",
        "a Canadian with deep subject understanding, using Canadian English expressions",
        "an advanced ESL student with comprehensive subject knowledge",

        # Expert
        "a Scottish subject expert integrating Scots dialect",
        "a non-native English speaker who is a leading subject expert",
        "an ESL professor teaching the subject at a postgraduate level with high English proficiency",
        "a native English speaker who is a renowned subject expert",
        "a native English speaker from Ireland who is a field leader, using Irish English idioms"
    ]

    for category in sample_categories:
        for style in styles:
            for role in roles:
                # we use JSON array for the later possibility of creating multiple answers per prompt
                prompt = f"""
As {role}, develop an answer for the question '{question.question}' in a maximum of two sentences.
Use a {style} as communication style.
Please return your answer as a JSON array: ["answer"]"""

                if question.sample_answer is not None:
                    prompt += f"\nConsider '{question.sample_answer}' as sample solution containing all relevant aspects."

                prompt += f"\nAlso {_setup_category(category, key_elements)}"

            yield prompt

def _find_json_content(choice:str):
    # remove new lines
    answer = re.sub(r'\n', ' ', choice.strip())
    if "[" not in answer and "]" not in answer:
        answer = f"[{answer}]"
    
    return re.search(r'\[(.*?)\]', answer)

def _generate_answers(question:Question, prompt) -> List[Answer]:
    prompt_size = _count_token(prompt)
    if prompt_size > 1000:
        log.warning(f"The prompt is very huge: {prompt_size}")

    # Set up parameters for generating answers
    max_tokens = 4000 - prompt_size
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

    for choice in generated_answers.choices:
        log.debug(f"generated answer: {choice.text}")
        contains_json = _find_json_content(choice.text)

        if contains_json:
            answer_str = contains_json.group(0)
            json_answers = json.loads(answer_str)
            return [Answer(question.question_id, answer.strip(), hashlib.md5(answer.strip().encode()).hexdigest(), -1) for answer in json_answers]

        raise JSONDecodeError(f"No valid JSON found in answer: {choice.text}")

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
        answer['answer'] = formatted_answer.strip()
        answer['rating'] = rating
    else:
        answer['rating'] = _validate_rating(answer['rating'])

    return True


def _validate_rating(rating):
    try:
        rating = int(rating)
        if rating >= 0 and rating <= 3:
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
            if parsed_number >= 0 and parsed_number <= 3:
                number = parsed_number
            break

    return final_answer, number

def generate_rating_prompt(question, numerated_answers, key_elements):
    prompt = f"""In this task, you will assess answers to a specific question, based on the presence of distinct key elements.
These elements may not be quoted verbatim, but their central meaning should be clearly conveyed in the response."""
    prompt = _add_key_elements(prompt, key_elements)

    prompt += f"""
You will classify each answer into categories, depending on the number of key elements it contains:
    Category 0: The answer includes none of the key elements.
    Category 1: The describes includes one key element.
    Category 2: The describes includes two key elements.
    Category 3: The describes includes three or more key elements.

Keep in mind, the punctuation, stylistic choices, or the specific wording used in an answer do not influence its score.
The evaluation is solely based on the presence or absence of the key elements.

The answers will be provided to you in JSON format, such as {{"answer_id1":"answer1"}}.
After you assess them, you should provide the scores in a similar JSON format: {{"answer_id1":"rating_id1"}}.

Question: "{question.question}"
Answers: "{numerated_answers}"
"""
    return prompt

def _rate_answers(api_key, question:Question, answers: Iterator[Answer], ignore_text_syntax, key_elements:List[KeyElement]) -> List[Answer]:
    openai.api_key = api_key
    numerated_rated_answers = {
        f"{idx+1}": answer for idx, answer in enumerate(answers)}

    # map {id:answer} sent to openAi
    numerated_answers = {f"{idx}": answer.answer
                         for idx, answer in numerated_rated_answers.items()}

    prompt = generate_rating_prompt(question, numerated_answers, ignore_text_syntax, key_elements)

    # Set up parameters for generating answers
    max_tokens = 10 * len(answers) # < 10 token / answer {"id":"rating"}
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
    for choice in generated_answers.choices:
        log.debug(f"generated rating: {choice.text}")
        # remove new lines
        answer = re.sub(r'\n', ' ', choice.text.strip())
        # find json content
        contains_json = re.search(r'\{.*\}', answer)

        if contains_json:
            answer_str = contains_json.group(0).replace("'", '"')
            json_answer = json.loads(answer_str)
            return [_add_rating(numerated_rated_answers, answer_id, rating_id) for answer_id, rating_id in json_answer.items()]
        raise JSONDecodeError(f"No valid JSON found in answer: {answer}")

def rate_answers(api_key, question:Question, answers: Iterator[Answer], key_elements:List[KeyElement]) -> Generator[List[Answer], None, None]:
    retry = 0
    # TODO: configuration
    max_retries = 5
    while retry < max_retries:
        try:
            yield _rate_answers(api_key, question, answers, key_elements)
            break
        except JSONDecodeError as e:
            retry += 1
            log.error(
                f"Unable to parse JSON response: {e}")
        except Exception as e:
            retry += 1
            log.error(
                f"Unable to rate Answer: {e}")

    if retry >= max_retries: 
        log.error(f"Exceeded {retry} retries, the rating of answers will be aborted")

        

def _add_rating(numerated_rated_answers, answer_id, rating_id):
    original_answer = numerated_rated_answers[answer_id]
    original_answer.score_2 = _validate_rating(rating_id)
    return original_answer