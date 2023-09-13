from tsv_utils import get_questions, get_key_elements_by_question_id, get_answers_per_question
from generate_base_answers import generate_answer_prompt, generate_rating_prompt
from config import Configuration
import random

import logging as log
from config_logger import config_logger

if __name__ == "__main__":
    config_logger(log.DEBUG, 'prompt.log')
    config = Configuration()
    key_elements_per_question = get_key_elements_by_question_id(config.get_key_elements_path())
    answers_per_question = get_answers_per_question(config.get_samples_path("davinci"))

    # We target 4k Answers per Question in total
    for question in get_questions(config.get_questions_path(), False):
        key_elements = key_elements_per_question[question.question_id]
        answers = random.sample(answers_per_question[question.question_id], 10)

        numerated_rated_answers = {
        f"{idx+1}": answer for idx, answer in enumerate(answers)}

        # map {id:answer} sent to openAi
        numerated_answers = {f"{idx}": answer.answer
                            for idx, answer in numerated_rated_answers.items()}
        
        print("Create answer Prompts:")
        print("##########################################################################################\n")
        for prompt in generate_answer_prompt(question, key_elements):
            print(prompt)
            print("##########################################################################################\n")
        
        rate_prompt = generate_rating_prompt(question, numerated_answers, key_elements)
        print(f"Rate answers Prompt:\n{rate_prompt}")
        print("##########################################################################################\n")
    