from typing import List, Iterator, Generator, Dict, Union, Any
from json.decoder import JSONDecodeError
import openai
import time
from openai.error import OpenAIError, RateLimitError
import re
import json
import logging as log
import re
from tsv_utils import Answer, Question, KeyElement
import hashlib
import tiktoken
import random
from enum import Enum
from open_ai_utils_davinci import SampleGoal, SampleRole


class CHAT_GPT_MODEL(Enum):
    """ Enumeration defining the GPT models to be used.

    This enum class lists the possible models that can be used. Each member represents a different GPT model variant.

    Attributes:
        TURBO (str): Represents the 'gpt-3.5-turbo' model variant. 
                     Note: This model may encounter difficulties in producing incorrect answers.
        GPT4 (str): Represents the 'gpt4' model variant.
    """ 
    TURBO = "gpt-3.5-turbo"
    GPT4 = "gpt4"


def _count_token(messages:List[Dict], model:CHAT_GPT_MODEL) -> int:
    """ Evaluates the number of token for a given messages

    Args:
        messages (str): The messages to be evaluated
        model (CHAT_GPT_MODEL): The model to be used

    Returns:
        int: Number of token
    """
    enc = tiktoken.encoding_for_model(model.value)
    return sum([len(enc.encode(message["content"])) for message in messages])

def generate_samples(api_key:str, question:Question, key_elements:List[KeyElement], model:CHAT_GPT_MODEL) -> Generator[List[Answer], None, None]:
    """ Generate unannotated sample answers for a given question using the Open AI API.

    This function connects to the Open AI API using the provided key and produces answers for the specified 
    question, taking into account the associated key elements from the respective ASAP EssaySet.

    Args:
        api_key (str): The API key for accessing Open AI's services
        question (Question): The question for which sample answers are to be generated
        key_elements (List[KeyElement]): List of key elements associated with the question according to the 
                                         respective ASAP EssaySet
        model (CHAT_GPT_MODEL): The model to be used

    Yields:
        Generator[List[Answer], None, None]: A generator that produces lists of generated answers. Each iteration yields
                                             a list of answers corresponding to the given question
    """
    openai.api_key = api_key

    for messages in generate_answer_messages(question, key_elements):
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                yield _generate_sample(
                        question, messages, model)
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
            log.error(f"Exceeded {max_retries} retries, the creation of answers for the question {question.question_id} will be aborted")

def _execute_api_call(messages:str, max_tokens:int, temperature:float, frequency_penalty:float, presence_penalty:float, model:CHAT_GPT_MODEL) -> Any:
    """ Executes a set of massages against the OpenAI chat completion API.

    This private method makes a call to the OpenAI chat completion API using the specified parameters. In the event of 
    `RateLimitError` or `OpenAIError`, the function will pause for a brief period and then retry the request. The 
    method will make up to a predetermined number of retries before logging an error and terminating

    Args:
        messages (str): The prompt to be executed by the OpenAI model
        max_tokens (int): Maximum number of tokens (words/characters) to be generated in the response
        temperature (float): Determines randomness in the AI's responses. Higher values make the output more random, 
                             while lower values make it more deterministic.
        frequency_penalty (float): Adjusts the likelihood of the AI generating frequent tokens
        presence_penalty (float): Adjusts the likelihood of the AI generating tokens based on their presence in the prompt
        model (CHAT_GPT_MODEL): The model to be used

    Raises:
        Exception: Any unexpected error encountered during the API call. Specific exceptions like `RateLimitError` 
                   and `OpenAIError` are handled internally with retries, but all others are raised

    Returns:
        Any: The response from the OpenAI API, typically containing the model's generated text
    """
    retries = 0
    max_retries = 3600
    while retries < max_retries:
        try:
            return openai.ChatCompletion.create(
                model=model.value,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=1,
                stop=None
            )
        except RateLimitError as e:
            retries += 1
            sleep_duration = 10
            log.warning(f"Rate limit hit: {str(e)}. Sleeping for {sleep_duration} before starting retry number: {retries}")
            time.sleep(sleep_duration)
        except OpenAIError as e:
            retries += 1
            sleep_duration = 10
            log.warning(f"Received OpenAIError: {str(e)}. Sleeping for {sleep_duration} before starting retry number: {retries}")
            time.sleep(sleep_duration)
        except Exception as e:
            log.error(f"An Exception ocurred while calling OpenAI: {str(e)}, Aborting the process.")
            raise e

    if retries >= max_retries:
        log.error(f"To many retries")


def _get_key_elements(key_elements:List[KeyElement]) -> str:
    """ Produces a String containing all given key elements as listing

    Args:
        key_elements (List[KeyElement]): The key elements to be included

    Returns:
        str: Listing of key elements
    """
    if len(key_elements) > 0:
        result = "The key elements for this task are:"
        for element in key_elements:
            result += f"\n- {element.element}"
    return result


def _setup_goal(goal:SampleGoal, key_elements:List[KeyElement]) -> str:
    """ Combines a defined SampleGoal with randomly selected key elements and returns the combination as a formatted string.

    This private method starts with the task description from the `SampleGoal` and then appends a defined number 
    of key elements chosen randomly from the provided list. The combined string is formatted with the task 
    description followed by the selected key elements listed one per line.

    Args:
        goal (SampleGoal): The SampleGoal object containing the task description and the number of required key elements
        key_elements (List[KeyElement]): A list of potential key elements to choose from

    Returns:
        str: A string combining the goal's task description with the selected key elements. If key elements 
             are selected, they are listed one per line following the task description
    """
    line = f"{goal.task}"
    random_elements = random.sample(key_elements, goal.number_of_key_elements)
    if len(random_elements) > 0:
        line += ":"
        for element in random_elements:
            line += f"\n- {element.element}"
    return line


def _get_idioms(idioms:Dict[str, List[str]], idiom_category:str) -> str:
    """ Retrieves a set of idioms from a specified category and formats them as single String.

    This private method randomly selects a predefined number of idioms from the given dictionary under the provided 
    category. If the category is not present in the dictionary or is `None`, an empty string is returned. 
    The selected idioms are then formatted as a list in the returned string.

    Args:
        idioms (Dict[str, List[str]]): A dictionary mapping categories of idioms to lists of idioms
        idiom_category (str):The category from which to select idioms

    Returns:
        str: A formatted string with a list of randomly chosen idioms from the specified category. 
             If the category is not found or is `None`, returns an empty string
    """
    if idiom_category is None or idiom_category not in idioms:
        return ""
    
    selection = random.sample(idioms[idiom_category], 3)
    result = "Use idioms such as:"
    for idiom in selection:
        result += f"\n- {idiom}"
    return result


# The wording has been enhanced with the assistance of Chat-GPT, optimizing their comprehensibility for text-davinci-003.
# The elements use a simple phrasing to make them easily understandable
def generate_answer_messages(question:Question, key_elements:List[KeyElement]) -> Generator[str, None, None]:
    """ Generates a series of 4200 AI message sets based on the question, key elements, and predefined configurations.

    This function produces a generator of message sets tailored for the OpenAI model. It incorporates several instructional 
    components such as role, task, style, and format, in the prompt, to guide the AI in generating desired responses. 
    The prompts also have the flexibility to embody various roles, styles, and goals which increases the diversity of 
    potential answers.

    Args:
        question (Question): The primary question for which the AI needs to generate answers
        key_elements (List[KeyElement]): List of key elements associated with the question according to the 
                                         respective ASAP EssaySet

    Yields:
        Generator[str, None, None]: A generator that yields the individual AI prompt instructions
    
    Notes:
        - The function integrates a vast array of instructions, from idioms, roles, goals to styles to provide varied 
          AI messages.
        - Certain combinations might seem contradictory, but the focus is more on language-style generation by the AI 
          rather than the logical coherence of the messages.
    """
    # 10 categories of correctness defined as goals within the prompt
    # NOTE: it does not work to provide a list and instruct the AI to choose a number of entries -> we need to provide just the entries
    # NOTE: it does not work to provide a list and instruct the AI to avoid any of its entries.
    sample_goals = [
        # the first two were only used with gpt-3.5-turbo for demonstrative purposes
        # SampleGoal(0, "Imagine you're playing a character in a play who always gets facts wrong and strays off-topic. How would that character answer the following?"),
        # SampleGoal(0, "Suppose you had the opposite of your training data — filled with incorrect facts and tangential thoughts. From that perspective, how would you respond to this?"),
        SampleGoal(0, "Produce the example of an entirely incorrect and off-topic answer"),
        SampleGoal(0, "Produce the example of an entirely incorrect answer that contradicts established scientific understanding."),
        SampleGoal(0, "The answer must be an analogy or metaphor to demonstrate comprehension."),
        SampleGoal(1 + question.score_offset, "The answer must include a paraphrase of all of the following information in the response."),
        SampleGoal(2 + question.score_offset, "The answer must include a paraphrase of all of the following information in the response."),
        SampleGoal(3 + question.score_offset, "The answer must include a paraphrase of all of the following information in the response."),
        SampleGoal(3 + question.score_offset, "The answer must include a paraphrase of all of the following information in an unrelated or incorrect context in the response."),
        SampleGoal(random.randint(3 + question.score_offset, len(key_elements)), "The answer must include a paraphrase of all of the following information in the response."),
        SampleGoal(random.randint(1, len(key_elements)), "The answer must include a paraphrase of all of the following information but deviate from the main topic in the response."),
        SampleGoal(len(key_elements), "The answer must contradict any of the following information, with reasoning.")
    ]

    # 21 combinations of tone syntax, mood, voice and style
    # This could be broken down in tones ["Neutral", ...] 
    #   and syntaxes ["complex scientific jargon", ...] 
    #   to further increase the number of variations of styles

    # NOTE: The AI is unable to follow to many instructions properly
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

    # this is less about accuracy and more about the incorporations of idioms in general
    idioms = {
        "category_1": [
            "All hat, no cattle",  # All talk and no action
            "As useful as a chocolate teapot",  # Useless
            "A hard row to hoe",  # A difficult task
            "That dog won't hunt",  # That idea won't work
            "Slower than molasses in January",  # Very slow
            "Like two peas in a pod",  # Very similar
            "Cute as a button",  # Very cute
            "Barking up the wrong tree",  # Misdirected efforts
            "Like finding a needle in a haystack",  # Extremely hard to find
            "Faster than a one-legged man in a butt-kicking contest"  # Very fast
        ],
        "category_2": [
            "Bite the bullet",  # Face a difficult situation
            "Hit the nail on the head",  # Exactly right
            "When pigs fly",  # Never
            "You can't judge a book by its cover",  # Don't prejudge
            "Once in a blue moon",  # Very rarely
            "You can't make an omelette without breaking eggs",  # It's impossible to achieve something important without causing some problems
            "Let the cat out of the bag", # Reveal a secret
            "Kill two birds with one stone", # Achieve two tasks simultaneously
            "Cutting corners" # Skipping steps to save time
        ],
        "category_3": [
            "Sweet nanny goat a go run him belly",  # Getting too much of a good thing can lead to trouble
            "Every mikkle mek a mukkle",  # Every little bit counts
            "Stone under water nuh know when sun hot" # Those isolated from changes remain unaware of them.
            "Chicken merry, hawk deh near" # Every action has a reaction, often seen in physics.
            "Ripe fruit must drop" # Natural processes follow a certain order.
            "When the river is silent, it’s either dried up or it’s becoming a flood" # An apparent lack of change can indicate stability or impending drastic change.
            "New broom sweeps clean, but old broom knows every corner" # While new techniques may appear superior, traditional methods often have proven effectiveness based on extensive experience.
            "One hand can't clap",  # It takes cooperation to achieve a task
            "Every day bucket a go well, one day the bottom must drop out" # Systems that are stressed continuously will eventually fail.
            "Frog say, what is joke to children is death to him" # The impact of actions can vary dramatically depending on perspective.
        ],
        "category_4": [
            "Up and at 'em",  # Get started
            "Colder than a witch's teat in a brass bra",  # Very cold
            "Off like a herd of turtles",  # Starting very slowly
            "In a coon's age",  # In a long time
            "Does a one-legged duck swim in circles?",  # Obvious answer is yes
            "Like herding cats",  # Trying to control an uncontrollable situation
            "Faster than a jackrabbit on a date",  # Very fast
            "Tight as bark on a tree",  # Stingy or frugal
            "Finer than frog hair",  # Very fine or delicate
            "Like trying to put lipstick on a pig"  # Trying to make something unattractive look attractive
        ],
        "category_5": [
            "Nae wind, nae wave", # No influence, no effects
            "Mony a mickle maks a muckle", # Many small things accumulate to something big
            "It's mince",  # It's rubbish or nonsense    
            "Cold enough to freeze the balls off a brass monkey", # Extremely cold
            "As damp as a dungeon", # Extremely wet or humid
            "As thin as a rake", # Extremely thin or slender
            "As high as a kite", # Very high
            "Heavy as a lead balloon", # Extremely heavy
            "Round as a pease", # Perfectly round
            "Strong as a lion" # Extremely strong
        ],
        "category_6": [
            "Colder than a witch's tit", # Extremely cold
            "Smaller than a clam's hindquarters", # Extremely small
            "Faster than a car on the Pike", # Very fast
            "Higher than Hancock tower", # Very high
            "Sharper than a Sox fan's wit", # Very sharp
            "Denser than chowder", # Very dense or thick
            "Slower than molasses in January", # Very slow
            "Stronger than a Southie dockworker", # Very strong
            "Hotter than a T platform in August", # Extremely hot
            "Quieter than a midnight in the Commons" # Very quiet
        ],
        "category_7": [
            "A leopard doesn't change its spots", # Immutable properties
            "Small-small", # Miniscule or incremental
            "Veld fire", # Rapid uncontrollable reaction
            "Cold as a Jo'burg morning", # Very cold
            "High as the Drakensberg", # Very high
            "Dry as the Karoo", # Very dry or arid
            "Quick-quick", # Quickly, swiftly
            "Slow as a wet week", # Very slow
            "Strong as a lion", # Very strong
            "Light as a feather", # Very light
        ],
        "category_8": [
            "Bigger than a prairie sky", # Extremely large or expansive
            "Cold as a Yukon winter", # Extremely cold or chilling
            "Swift as a Calgary wind", # Very fast or rapid
            "As changeable as Maritime weather", # Highly variable or changeable
            "Solid as Canadian Shield", # Extremely sturdy or stable
            "Heavy as a moose", # Extremely heavy or substantial
            "Steady as a Canuck's resolve", # Extremely steady or stable
            "Twisted as Toronto's streets", # Complicated or convoluted
            "Hot as Toronto in July", # Extremely hot or sweltering
            "Tight as a beaver's dam" # Extremely compact or close-fitted
        ],
        "category_9": [
            "Arseways",  # To do something the wrong way
            "Bang on",  # Correct, right
            "Donkey's years",  # A long time
            "On the never never",  # Buying on hire purchase
            "Puck",  # To hit, punch or thump something
            "Quare",  # Very, extremely
            "Like hen's teeth", # Extremely rare
            "Wet as an otter's pocket", # Extremely wet
            "As fast as greased lightning", # Extremely fast
            "As solid as the Rock of Cashel" # Very solid/stable
        ]
    }

    # 20 roles the AI embodies by creating the answers
    # The AI is unable to just incorporate idioms, we need to provide examples
    roles = [
        # Beginner
        SampleRole("an American Southerner using regional vernacular, new to the subject", "category_1"),
        SampleRole("an ESL student with basic English skills, discovering the subject", "category_2"),
        SampleRole("a Caribbean native with beginner English skills, learning the subject with Creole influences", "category_3"),
        SampleRole("an English-speaking student learning the topic for the first time", None),
        SampleRole("a non-native English speaker with limited English, navigating basic concepts of the topic", "category_2"),
        
        # Intermediate
        SampleRole("an American Midwesterner with intermediate subject knowledge, using regional dialect", "category_4"),
        SampleRole("a Scot with foundational subject knowledge, integrating Scottish slang", "category_5"),
        SampleRole("an British English speaker studying the topic at an intermediate level", None),
        SampleRole("an undergraduate student with a basic grasp on the subject", None),
        SampleRole("a non-specialist approaching the subject from a different field's perspective", None),

        # Advanced
        SampleRole("a british English speaking senior researcher with considerable subject knowledge", None),
        SampleRole("a Bostonian with substantial understanding of the subject, using local dialect", "category_6"),
        SampleRole("a South African with significant subject understanding, integrating local colloquialisms", "category_7"),
        SampleRole("a Canadian with deep subject understanding, using Canadian English expressions", "category_8"),
        SampleRole("an advanced ESL student with comprehensive subject knowledge", None),

        # Expert
        SampleRole("a Scottish subject expert integrating Scots dialect", "category_5"),
        SampleRole("a non-native English speaker who is a leading subject expert", None),
        SampleRole("an ESL professor teaching the subject at a postgraduate level with high English proficiency", None),
        SampleRole("a british English speaker who is a renowned subject expert", None),
        SampleRole("a native English speaker from Ireland who is a field leader, using Irish English idioms", "category_9")
    ]

    for goal in sample_goals:
        for style in styles:
            for role in roles:
                # JSON allows to avoid phrases like 'I apologize for any confusion. Here is the correct answer:  '
                # the prompt is categorized to simplify its interpretation using xml-ish syntax
                # the Segments need to be enclosed, it has been observe that the AI just extends the prompt, changing the result
                messages = [
                    {"role": "system", "content": f"You are {role.role} producing sample answers to train an AI."},
                    {"role": "user", "content": f"Please develop an answer for the question '{question.question}'."},
                    {"role": "user", "content": f"{_get_idioms(idioms, role.idiom_category)}"},
                    {"role": "user", "content": f"Write your answer in a {style} as communication style."},
                    {"role": "user", "content": f"The answer should not be longer than two sentences."},
                    {"role": "user", "content": f"Just provide the answer without any additional text."},
                    #{"role": "user", "content": f"Please return your answer as a JSON array: [\"answer\"]"}
                ]

                if question.sample_answer is not None:
                    messages.append({"role": "user", "content": f"Consider '{question.sample_answer}' as sample solution containing all relevant aspects."})

                messages.append({"role": "user", "content": f"{_setup_goal(goal, key_elements)}"})
                yield messages


def _generate_sample(question:Question, messages:List[Dict[str, str]], model:CHAT_GPT_MODEL) -> List[Answer]:
    """ Generates an AI-crafted answer for a given question based on the specified messages.

    The function attempts to produce a single answer by leveraging the OpenAI model. These answer is 
    extracted from the generated content assuming it is provided as plain text. The 
    function also ensures that the total token count (prompt + answer) does not exceed a set limit

    Args:
        question (Question): The primary question for which the AI needs to generate answers
        prompt (str): The OpenAI prompt that instructs the model on how to answer the question
        model (CHAT_GPT_MODEL): The model to be used

    Returns:
        List[Answer]: A list containing one generated (unannotated) answer. The answer is represented 
                      as an `Answer` object with various attributes such as the question Id,
                      the answer text, its unique hash, etc.
    """
    messages_token_count = _count_token(messages, model)
    if messages_token_count > 1000:
        log.warning(f"The prompt is very huge: {messages_token_count}")
    
    log.debug(f"Create sample answers with the following prompt: {messages}")

    # Set up parameters for generating answers
    max_tokens = 4000 - messages_token_count
    temperature = 0.8
    frequency_penalty = 0.2
    presence_penalty = 0.6

    # Generate answers
    generated_answers = _execute_api_call(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        model=model
    )

    for choice in generated_answers.choices:
        log.debug(f"generated choice: {choice}")

        message = choice.message['content']
        log.info(f"generated answer: {message}")
        answer = re.sub('\n+', ' ', message).strip()
        return [Answer(question.question_id, answer, hashlib.md5(answer.encode()).hexdigest(), -1, -1)]


def _validate_rating(rating:Union[str, int]) -> int:
    """ Validates and corrects the rating value.

    This function ensures that the rating adheres to the valid range [0, 3]. 
    Ratings below 0 are set to 0, assuming no key elements could be detected. 
    Ratings above 3 are corrected to 3, assuming more than the required amount 
    of key elements for this rating were present. If the rating cannot be parsed 
    or validated, it returns -1.
    Args:
        rating (Union[str, int]): The rating to be validated, either as a string or integer

    Returns:
        int: The corrected and validated rating value
    """
    error_msg = ""
    try:
        rating = int(rating)
        if rating < 0:
            log.error(f"Identified invalid rating: {rating}.")
            return 0
        
        if rating > 3:
            log.error(f"Identified invalid rating: {rating}.")
            return 3

        if rating >= 0 and rating <= 3:
            return rating

    except Exception as e:
        error_msg = f" Error: {str(e)}"

    log.error(f"Identified invalid rating: {rating}. {error_msg}")
    return -1


def _get_scale_element_for(score:int, required_key_elements:int) -> str:
    """ Retrieves a rating scale string based on a given score and the number of key elements.

    This function returns a descriptive string that indicates the number of key elements 
    associated with a particular score. For instance, if the score is 2 and the required number 
    of key elements is 1, the function will return "2: The answer includes one of the key element."

    Args:
        score (int): The designated score
        required_key_elements (int): The count of key elements corresponding to the score

    Raises:
        ValueError: If `required_key_elements` is negative

    Returns:
        str: A formatted string describing the relation between the score and the key elements
    """
    if required_key_elements < 0:
        raise ValueError("required_key_elements cannot be negative.")
    if required_key_elements == 0:
        return f"{score}: The answer includes none of the key elements."
    if required_key_elements == 1:
        return f"{score}: The answer includes one of the key element."
    if required_key_elements >= 1:
        return f"{score}: The answer includes {required_key_elements} key elements."
    

# We rate multiple answers at once, this is supposed to make ratings more consistent
def generate_annotation_messages(question:Question, numerated_answers:Dict[str, str], key_elements:List[KeyElement]) -> str:
    """ Constructs the messages directing the OpenAI model to evaluate a set of answers based on the presence of specific key elements.

    The function assembles messages for the OpenAI model to grade sample answers according to the presence or absence of 
    predefined key elements as defined by the respective ASAP EssaySet. The evaluation score ranges from 0 to 3, with the score 
    determined by how many key elements an answer contains.

    Args:
        question (Question): The question for which sample answers are to be rated
        numerated_answers (Dict[str, str]): A dictionary of answers and their corresponding Ids
        key_elements (List[KeyElement]): List of key elements associated with the question according to the 
                                         respective ASAP EssaySet

    Returns:
        str: The fully constructed set of AI messages, detailing evaluation criteria and instructions for providing scores in JSON format
    """
    return [
        {"role": "system", "content": f"You are expert that assess answers to a specific question, based on the presence of distinct key elements"},
        {"role": "user", "content": f"These elements may not be quoted verbatim, but their central meaning should be clearly conveyed in the response."},
        {"role": "user", "content": _get_key_elements(key_elements)},
        {"role": "user", "content": f"""You will classify each answer into categories, depending on the number of key elements it contains from 0 to 3:
    {_get_scale_element_for(0, 0 + question.score_offset)}
    {_get_scale_element_for(1, 1 + question.score_offset)}
    {_get_scale_element_for(2, 2 + question.score_offset)}
    {_get_scale_element_for(3, 3 + question.score_offset)}"""},
        {"role": "user", "content": "Keep in mind, the punctuation, stylistic choices, or the specific wording used in an answer do not influence its score."},
        {"role": "user", "content": "The evaluation is solely based on the presence or absence of the key elements."},
        {"role": "user", "content": "The answers will be provided to you in JSON format, such as {{\"answer_id1\":\"answer1\"}}."},
        {"role": "user", "content": "After you assess them, you should provide the scores in a similar JSON format: {{\"answer_id1\":\"category\"}}."},
        {"role": "user", "content": f"Question: \"{question.question}\""},
        {"role": "user", "content": f"Answers: \"{numerated_answers}\""}
    ]

def _annotate_samples(api_key, question:Question, answers: Iterator[Answer], key_elements:List[KeyElement], score_type:int, model:CHAT_GPT_MODEL) -> List[Answer]:
    """ Annotates sample answers according to the presence or absence of predefined key elements as defined by the respective ASAP EssaySet.
    
    The evaluation score ranges from 0 to 3, with the score determined by how many key elements an answer contains.

    Args:
        api_key (str): The API key for accessing Open AI's services
        question (Question): The question for which sample answers are to be rated
        answers (Iterator[Answer]): An iterator containing the answers that need to be rated
        key_elements (List[KeyElement]): List of key elements associated with the question according to the 
                                         respective ASAP EssaySet
        score_type (int): The score type to be set by the evaluation (either 1 or 2)
        model (CHAT_GPT_MODEL): The model to be used

    Raises:
        JSONDecodeError: An error raised if the response from the OpenAI model cannot be parsed as valid JSON

    Returns:
        List[Answer]: A list of answers that have been annotated with their respective scores
    """
    openai.api_key = api_key
    numerated_rated_answers = {
        f"{idx+1}": answer for idx, answer in enumerate(answers)}

    # map {id:answer} sent to openAi
    numerated_answers = {f"{idx}": answer.answer
                         for idx, answer in numerated_rated_answers.items()}

    messages = generate_annotation_messages(question, numerated_answers, key_elements)
    log.debug(f"Annotate sample answers with the following prompt: {messages}")

    # Set up parameters for generating answers
    max_tokens = 10 * len(answers) # < 10 token / answer {"id":"rating"}
    temperature = 0.4
    frequency_penalty = 0.6
    presence_penalty = 0.2

    annotated_answers = _execute_api_call(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        model=model
    )
    log.debug(f"Received the following response '{annotated_answers}'")

    # Extract answers from OpenAI API response
    for choice in annotated_answers.choices:
        log.debug(f"generated choice: {choice}")
        message = choice.message['content']

        # remove new lines
        answer = re.sub(r'\n', ' ', message.strip())
        # find json content
        contains_json = re.search(r'\{.*\}', answer)

        if contains_json:
            answer_str = contains_json.group(0).replace("'", '"')
            json_answer = json.loads(answer_str)
            return [_add_rating(numerated_rated_answers, answer_id, rating, score_type) for answer_id, rating in json_answer.items()]
        raise JSONDecodeError(f"No valid JSON found in answer: {answer}")

def annotate_samples(api_key:str, question:Question, answers:Iterator[Answer], key_elements:List[KeyElement], model:CHAT_GPT_MODEL) -> Generator[List[Answer], None, None]:
    """ Annotates sample answers for a given question using the Open AI API.

    This function connects to the Open AI API using the provided key and annotates answers for the specified 
    question, taking into account the associated key elements from the respective ASAP EssaySet.

    Args:
        api_key (str):The API key for accessing Open AI's services
        question (Question): The question for which sample answers are to be rated
        answers (Iterator[Answer]): An iterator containing the answers that need to be rated
        key_elements (List[KeyElement]): List of key elements associated with the question according to the 
                                         respective ASAP EssaySet
        model (CHAT_GPT_MODEL): The model to be used

    Yields:
        Generator[List[Answer], None, None]: A generator that produces lists of annotated answers. Each iteration yields
                                             a list of answers corresponding to the given question
    """
    retry = 0
    max_retries = 5
    while retry < max_retries:
        try:
            log.debug(f"Try to rate answers with retry number: {retry}")
            # rate score_1 and score_2
            rated_answers = _annotate_samples(api_key, question, answers, key_elements, 1, model)
            yield _annotate_samples(api_key, question, rated_answers, key_elements, 2, model)
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


def _add_rating(numerated_rated_answers:Dict[str, Answer], answer_id:str, rating:Union[str, int], score_type:int) -> Answer:
    """ Associates a provided rating with its corresponding answer based on the answer's Is and score type.

    This function updates the provided rating for a given answer Is and score type. The score type can either be 1 or 2, 
    representing two independent evaluations

    Args:
        numerated_rated_answers (Dict[str, Answer]): A dictionary mapping answers to their respective Ids
        answer_id (str): The unique identifier of the answer to be updated
        rating (Union[str, int]): The new rating to be added or updated for the answer
        score_type (int): The score type to be set by the evaluation (either 1 or 2)

    Returns:
        Answer: The annotated answer
    """
    original_answer = numerated_rated_answers[answer_id]
    log.debug(f"Validate rating: {rating} for answer_id: {answer_id} and score_type: {score_type}")
    if score_type == 1:
        original_answer.score_1 = _validate_rating(rating)
    else:
        original_answer.score_2 = _validate_rating(rating)

    return original_answer