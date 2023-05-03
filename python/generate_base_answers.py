from typing import List
import openai
import re
import json
import argparse
import logging as log

model_engine = "text-davinci-002"

def generate_answers(api_key, quantity_of_answers, ignore_text_syntax, question, solution, question_id):
    openai.api_key = api_key

    # We target 1k Answers per Question
    sampleCategories = [
        "The answers should describe all aspects according to the sample solution",
        "The answers should contain one wrong aspect",
        "The answers should miss at leased one aspect",
        "The answers should be entirely wrong"
    ]

    # TODO: add question_id
    answers = []
    for category in sampleCategories:
        answers = answers + __generate_answers(question, solution, ignore_text_syntax, category, quantity_of_answers)
    
    return answers

def __generate_answers(question, solution, ignore_text_syntax, task, quantity_of_answers=20) -> List[str]:
    # Define prompt
    basePrompt = f"""Create possible answers for a test. 
    The question '{question}' should be answered on {quantity_of_answers} different ways using up to 2 Sentences each. 
    Each answer should be rated from 0 to 100 % regarding its correctness.
    Present the answers and their ratings in a JSON format like [{{"answer":"answer1", "rating":"100%"}}]"""

    if solution is not None:
        basePrompt += f"\nConsider '{solution}' as sample solution containing all relevant aspects."

    if ignore_text_syntax:
        basePrompt += "\nIgnore spelling or punctuation mistakes for the evaluation."

    prompt = basePrompt + f"\n{task}"

    # Set up parameters for generating answers
    max_tokens = 1024
    temperature = 0.7
    frequency_penalty = 0.5
    presence_penalty = 0.5

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

        # remove new lines
        answer = re.sub(r'\n', ' ', choice.text.strip())
        # find json content
        contains_json = re.search(r'\[(.*?)\]', answer)

        if contains_json:
            answer_str = contains_json.group(0)
            log.debug(f"About to parse the following answer as JSON: {answer_str}")
            json_answer = json.loads(answer_str)
            answers.extend(json_answer)

    # Return list of answers
    return answers




if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="The question for which to generate answers")
    parser.add_argument('--answer', default=None, help='A sample answer (optional)')
    parser.add_argument('quantity', default=20, help='Amount of created Answers')
    parser.add_argument('api_key', help='The API key for the OpenAI API')
    parser.add_argument('--ignore_text_syntax', action='store_true', help='Ignore spelling or punctuation mistakes for the evaluation')
    args = parser.parse_args()

    # Print list of answers, the question_id is always '1' hence there is only one question
    for i, answer in enumerate(generate_answers(args.api_key, args.quantity, args.ignore_text_syntax, args.question, args.answer, 1)):
        print(f"{i+1}. {answer}")