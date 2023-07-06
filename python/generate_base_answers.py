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

def _count_token(prompt:str) -> int:
    # TODO: model to config
    #enc = tiktoken.encoding_for_model("gpt-4")
    enc = tiktoken.encoding_for_model("text-davinci-003")

    return len(enc.encode(prompt))

def generate_answers(api_key, batch_size, question:Question, key_elements:List[KeyElement]) -> Generator[List[Answer], None, None]:
    openai.api_key = api_key

    for prompt in generate_answer_prompt(question, batch_size, key_elements):
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
        result = prompt + "\n\nThe key elements are:"
        for element in key_elements:
            result += f"\n- {element.element}"
    return result + "\n"

# 10 x 20 x 3 = 600 batches.
# With a batch size of 3, 1800 Answers will be created during one execution
def generate_answer_prompt(question:Question, batch_size, key_elements:List[KeyElement]):

    # 10 categories of correctness
    sample_categories = [
        "The answers must not contain any key element expressed in different words",
        "The answers must contain exactly one key element expressed in different words",
        "The answers must contain exactly two key elements expressed in different words",
        "The answers must contain exactly three key elements expressed in different words",
        "The answers must contain exactly three key elements but in a wrong context",
        "The answers must contain at leased three key elements",
        "The answers must be off-topic but contain key elements",
        "The answers must be contradictory",
        "The answers must contain misinformation",
        "The answers must be wrong and less than 8 words in length."
    ]

    # 20 combinations of tone syntax, mood, voice and style
    styles = [
        "delivered with a satirical tone, employ complex syntax, establish a joyful mood, utilize a first-person voice, and maintain a descriptive style",
        "drafted in a formal tone, use simple syntax, cultivate a gloomy mood, adopt a third-person limited voice, and demonstrate an expository style",
        "composed with a condescending tone, incorporate compound syntax, evoke a suspenseful mood, employ a second-person voice, and adhere to a narrative style",
        "articulated in an affectionate tone, feature interrogative syntax, foster a melodramatic mood, rely on a stream-of-consciousness voice, and exemplify a persuasive style",
        "written with an energetic tone, use declarative syntax, generate an excited mood, be expressed in an epistolary voice, and follow an analytical style",
        "conveyed in a respectful tone, apply active syntax, nurture a nostalgic mood, utilize an autobiographical voice, and espouse an argumentative style",
        "formulated with a bitter tone, involve passive syntax, create an anxious mood, adopt a third-person omniscient voice, and uphold a cause-and-effect style",
        "expressed in a candid tone, incorporate fragmented syntax, set a romantic mood, use an unreliable voice, and embrace a compare-and-contrast style",
        "presented with an enthusiastic tone, feature inverted syntax, kindle a somber mood, apply a reliable voice, and maintain a problem-solution style",
        "developed in an indifferent tone, use elliptical syntax, engender an angry mood, deploy a subjective voice, and exemplify a sequential style",
        "written in a humorous tone, integrate cumulative syntax, foster a peaceful mood, employ an objective voice, and espouse a definition style",
        "shaped with a serious tone, utilize periodic syntax, invoke a fearful mood, use a naive voice, and adopt a classification style",
        "formulated in a whimsical tone, incorporate parallel syntax, elicit a mysterious mood, employ an experienced voice, and adhere to a process style",
        "constructed with a melancholic tone, apply antithetical syntax, generate a reflective mood, use a childlike voice, and uphold a critique style",
        "articulated in a sincere tone, integrate nominal syntax, stimulate a whimsical mood, utilize a worldly voice, and maintain a satirical style",
        "expressed with a pessimistic tone, employ verbal syntax, kindle an euphoric mood, adopt an introspective voice, and demonstrate an symbolic style",
        "composed in an optimistic tone, use balanced syntax, provoke a despairing mood, apply a detached voice, and embrace an allegorical style",
        "delivered in an ironic tone, incorporate parenthetical syntax, cultivate a hopeful mood, employ a cynical voice, and follow an ironical style",
        "drafted with an informal tone, apply imperative syntax, stir an inspiring mood, utilize an idealistic voice, and uphold a metaphorical style",
        "written in a sarcastic tone, feature exclamatory syntax, create a bizarre mood, use an innocent voice, and adhere to a realistic style"
    ]

    # 3 roles the AI embodies by creating the answers
    roles = [
        "student learning the topic for the first time",
        "researcher exploring novel aspects of the topic",
        "professor with a deep understanding of the subject"
    ]

    for category in sample_categories:
        for style in styles:
            for role in roles:
                prompt = f"""
Imagine you're a {role}.
Develop {batch_size} distinct responses for the question '{question.question}' Each response should be confined to a maximum of two sentences.
The text should be {style}.
Try to use different analogies, examples or contextual scenarios in your responses.
Represent the responses and their scores in a JSON array of strings, following this structure: ["answer1", "answer2"]"""

                if question.sample_answer is not None:
                    prompt += f"\nConsider '{question.sample_answer}' as sample solution containing all relevant aspects."

                prompt = _add_key_elements(prompt, key_elements)

                prompt += f"\nNOTE: {category}"

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

def generate_rating_prompt(question, numerated_answers, ignore_text_syntax, key_elements):
    prompt = f"""I am an AI trained to score responses based on a four-point scale of relevance, coherence, and completeness.
The answers to be rated are formatted as JSON in the following matter {{"answer_id1":"answer1"}} 
Evaluate the following answers based on the quantity of key elements present from 0 to 3:"""

    prompt = _add_key_elements(prompt, key_elements)

    prompt += f"""
Question: {question.question}
Answers: {numerated_answers}

rating_id: criteria
0: Other
1: One key element
2: Two key elements
3: Three key elements

Present the ratings in a JSON format like {{"answer_id1":"rating_id1"}}"""

    if ignore_text_syntax:
        prompt += "\nIgnore spelling or punctuation mistakes for the evaluation."
    
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

def rate_answers(api_key, question:Question, answers: Iterator[Answer], ignore_text_syntax, key_elements:List[KeyElement]) -> Generator[List[Answer], None, None]:
    retry = 0
    # TODO: configuration
    max_retries = 5
    while retry < max_retries:
        try:
            yield _rate_answers(api_key, question, answers, ignore_text_syntax, key_elements)
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