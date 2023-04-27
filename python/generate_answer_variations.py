from typing import List
import openai
import re
import argparse
import json

# Set up OpenAI API key and model
# TODO: -> parameter
openai.api_key = "INSERT_API_KEY_HERE"

# TODO: evaluate other (cheaper) models
model_engine = "text-davinci-002"

def generate_variations(question, answer, numberOfVariations=20) -> List[str]:
    # Define prompt
    prompt = f"""Using the provided answer {answer} and the associated question {question}, 
        create {numberOfVariations} unique rephrasings by altering the vocabulary, syntax, and tone. 
        Ensure that the meaning of the original answer is preserved, while introducing variations in the language used."""

    # Set up parameters for generating variation
    max_tokens = 1024
    temperature = 0.7
    frequency_penalty = 0.5
    presence_penalty = 0.5

    # Generate variations
    generated_variations = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        n=1,
        stop=None,
    )

    # Extract variations from OpenAI API response
    variations = []
    for choice in generated_variations.choices:
        variation = re.sub(r'\n', ' ', choice.text.strip())
        variations.append(variation)

    # Return list of variation
    return variations

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="the question for which to generate answers")
    args = parser.parse_args()

    # We target 1k Answers (including variations) per Question
    sampleCategories = [
        "Change Tone",
        "Change vocabulary",
        "Change syntax"
    ]
    # DEBUG: Describe in detail how you modified each answer and how you evaluated them in a list at the end.
    # Generate variations
    variations = []
    for category in sampleCategories:
        variations = variations + generate_variations(args.question, args.answer)

    # Print list of answers
    for i, variation in enumerate(variations):
        print(f"{i+1}. {variation}")