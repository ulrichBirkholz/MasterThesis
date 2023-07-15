from tsv_utils import KeyElement, Question
from generate_base_answers import generate_answer_prompt, generate_rating_prompt

import argparse
import logging as log

# Setup and parse arguments
def setup_args():
    parser = argparse.ArgumentParser(description='Print prompts as they will be sent to open.ai')
    
    # 2 produces stable results, 5 is unstable, so some responses are unparsable 10 and higher was unusable
    parser.add_argument('batch_size', default=3, type=int, help='Each of the 600 batches, generates "batch_size" number of answers. With a batch size of 3, this equates to 1,800 answers being produced.')
    parser.add_argument('--use_sample', action='store_true', help='Use SampleAnswer for more context')
    return parser.parse_args()

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)
    log.basicConfig(filename='prompt.log', filemode='w')

    args = setup_args()

    # We target 4k Answers per Question in total
    for question in [Question(f"Question-{id}", None, str(id)) for id in range(2)]:
        key_elements = [KeyElement(question.question_id, f"Key-element-{id}") for id in range(5)]
        
        print("Create answer Prompts:")
        print("##########################################################################################\n")
        for prompt in generate_answer_prompt(question, args.batch_size, key_elements):
            print(prompt)
            print("##########################################################################################\n")
        
        rate_prompt = generate_rating_prompt(question, {f"{idx}": answer
                         for idx, answer in enumerate(['answer 1', 'answer 2', 'answer 3'])}, True, key_elements)
        print(f"Rate answers Prompt:\n{rate_prompt}")
        print("##########################################################################################\n")
    