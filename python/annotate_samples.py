from tsv_utils import get_questions, write_answers_tsv, get_answers_per_question, get_key_elements_by_question_id, Answer, Question, KeyElement
from open_ai_utils_message import annotate_samples as annotate_samples_message
from open_ai_utils_prompt import annotate_samples as annotate_samples_prompt, CHAT_GPT_MODEL
from config import Configuration
from typing import Dict, List, Callable, Generator
import argparse
from argparse import Namespace
import logging as log
from config_logger import config_logger


def setup_args() -> Namespace:
    """ Setup of the execution arguments

    Returns:
        Namespace: arguments to be used
    """
    parser = argparse.ArgumentParser(description='Create annotated Answers')
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    # 2 produces stable results, 5 is unstable, so some responses are unparsable 10 and higher was unusable
    parser.add_argument('chunk_size', default=2, type=int, help='Maximal amount of simultaneously annotated Answers')
    return parser.parse_args()


def _chunk(answers:List[Answer], size:int) -> Generator[List[Answer], None, None]:
    """ Splits a list of answers into chunks of a specified size
    
    Breaks down the provided list of answers into smaller lists (chunks) each containing
    a number of items as defined by the 'size' parameter. If the total number of answers 
    isn't a multiple of 'size', the last chunk will contain the remaining items

    Args:
        answers (List[Answer]): List of answers to be divided into chunks
        size (int): Number of answers each chunk should contain

    Yields:
        List[Answer]: The next chunk of answers
    """
    for i in range(0, len(answers), size):
        yield answers[i:i + size]


def _process_samples(
        answers_per_question: Dict[str, Answer],
        questions: List[Question],
        key_elements_per_question: Dict[str, List[KeyElement]],
        answers_path: str,
        api_key: str,
        chunk_size: int,
        callback: Callable[[str, Question, Generator[List[Answer], None, None], List[KeyElement], CHAT_GPT_MODEL],
                           Generator[List[Answer], None, None]],
        model: CHAT_GPT_MODEL
) -> None:
    """ Annotates a given set of answers using the specified model and saves the results in a TSV file

    Iterates over the provided questions, breaks down the corresponding answers into chunks, and uses Open AI 
    to annotate them. The annotated results are then stored in a TSV file

    Args:
        answers_per_question (Dict[str, Answer]): Dictionary mapping question IDs to their corresponding answers
        questions (List[Question]): List of all possible questions
        key_elements_per_question (Dict[str, List[KeyElement]]): Dictionary mapping question IDs to the 
            key elements associated with the question as per the respective ASAP EssaySet
        answers_path (str): Path to the TSV file where the annotated answers will be saved
        api_key (str): API key for accessing OpenAI's services
        chunk_size (int): Size of each chunk when breaking down the list of answers for processing
        callback (Callable): A function that interacts with OpenAI to annotate the samples. The function 
            takes in the API key, question, chunk of answers, key elements, and the model, and returns 
            a generator producing the annotated answers
        model (CHAT_GPT_MODEL): The specific model to use for the annotation process
    """
    for question_id, answers in answers_per_question.items():
        key_elements = key_elements_per_question[question_id]
        
        # find question by id
        question = next(filter(lambda question: question.question_id == question_id, questions), None)
        if question:
            for chunk in _chunk(answers, chunk_size):
                write_answers_tsv(answers_path, callback(api_key, question, chunk, key_elements, model), True)

        else:
            log.error(f"no matching question found for Id: {question_id}")


if __name__ == "__main__":
    config_logger(log.DEBUG, 'annotate.log')

    args = setup_args()
    config = Configuration()
    questions = get_questions(config.get_questions_path(), False)
    key_elements_per_question = get_key_elements_by_question_id(config.get_key_elements_path())

    davinci_rating_expert_data = config.get_samples_path("davinci_rating_expert_data")
    davinci_rated_answer_path = config.get_samples_path("davinci")

    turbo_rating_expert_data = config.get_samples_path("turbo_rating_expert_data")
    turbo_rated_answer_path = config.get_samples_path("turbo")

    gpt4_rating_expert_data = config.get_samples_path("gpt4_rating_expert_data")
    gpt4_rated_answer_path = config.get_samples_path("gpt4")

    unrated_davinci_answers_per_question = get_answers_per_question(config.get_unrated_samples_path("davinci"))
    unrated_gpt4_answers_per_question = get_answers_per_question(config.get_unrated_samples_path("gpt4"))
    expert_answers_per_question = get_answers_per_question(config.get_samples_path("experts"))

    _process_samples(unrated_davinci_answers_per_question, questions, key_elements_per_question, davinci_rated_answer_path, args.api_key, args.chunk_size, annotate_samples_prompt, CHAT_GPT_MODEL.DAVINCI_003)
    _process_samples(expert_answers_per_question, questions, key_elements_per_question, davinci_rating_expert_data, args.api_key, args.chunk_size, annotate_samples_prompt, CHAT_GPT_MODEL.DAVINCI_003)
    
    _process_samples(unrated_davinci_answers_per_question, questions, key_elements_per_question, turbo_rated_answer_path, args.api_key, args.chunk_size, annotate_samples_message, CHAT_GPT_MODEL.TURBO)
    _process_samples(expert_answers_per_question, questions, key_elements_per_question, turbo_rating_expert_data, args.api_key, args.chunk_size, annotate_samples_message, CHAT_GPT_MODEL.TURBO)

    _process_samples(unrated_gpt4_answers_per_question, questions, key_elements_per_question, gpt4_rated_answer_path, args.api_key, args.chunk_size, annotate_samples_message, CHAT_GPT_MODEL.GPT4)
    _process_samples(expert_answers_per_question, questions, key_elements_per_question, gpt4_rating_expert_data, args.api_key, args.chunk_size, annotate_samples_message, CHAT_GPT_MODEL.GPT4)
