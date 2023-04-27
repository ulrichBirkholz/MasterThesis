from typing import List
import openai
import re
import argparse
import json

# Set up OpenAI API key and model
# TODO: -> parameter
openai.api_key = "INSERT_API_KEY_HERE"
model_engine = "text-davinci-002"

def generate_answers(question, solution, ignoreTextSyntax, task, numberOfSamples=20) -> List[str]:
    # Define prompt
    basePrompt = f"""Create possible answers for a test. 
    The question '{question}' should be answered on {numberOfSamples} different ways using up to 2 Sentences each. 
    Each answer should be rated from 0 to 100 % regarding its correctness."""

    if solution is not None:
        basePrompt += f"\nConsider '{solution}' as sample solution containing all relevant aspects."

    if ignoreTextSyntax:
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
        answer = re.sub(r'\n', ' ', choice.text.strip())
        answers.append(answer)

    # Return list of answers
    return answers

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="the question for which to generate answers")
    args = parser.parse_args()

    # We target 1k Answers per Question
    sampleCategories = [
        "The answers should describe all aspects according to the sample solution",
        "The answers should contain one wrong aspect",
        "The answers should miss at leased one aspect",
        "The answers should be entirely wrong"
    ]
    # DEBUG: Describe in detail how you modified each answer and how you evaluated them in a list at the end.
    # Generate possible answers
    answers = []
    for category in sampleCategories:
        answers = answers + generate_answers(args.question, args.solution, args.ignoreTextSyntax, category)

    # Print list of answers
    for i, answer in enumerate(answers):
        print(f"{i+1}. {answer}")