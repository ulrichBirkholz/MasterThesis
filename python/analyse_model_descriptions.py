import os
import json
from config import Configuration
import logging as log
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import argparse
from argparse import Namespace


def _tabs_for_alignment(text:str) -> str:
    """ Adds a specific number of tabs to align text based on its length

    Args:
        text (str): The input text whose length is to be evaluated

    Returns:
        str: A string of tabs. The number of tabs added is determined by the length of the input text
    """
    if len(text) >= 32:
        tabs = 8
    else:
        tabs = 1
    return "\t" * tabs


def _generate_model_id(version:str, question_id:str, data_source:str) -> str:
    """ Generates a unique identifier for a model using its version, associated essay set Id, and training data source

    Args:
        version (str): Internal identifier representing the model type and version
        question_id (str): The Id of the associated Essay Set
        data_source (str): The origin or source of the training data for the model

    Returns:
        str: A concatenated string representing the model's unique identifier
    """
    return f"{version}_{question_id}{data_source}"



# Setup and parse arguments
# example: python -m analyse_model_descriptions
def setup_args() -> Namespace:
    """ Setup of the execution arguments

    Returns:
        Namespace: arguments to be used
    """
    parser = argparse.ArgumentParser(description='Evaluate model descriptors and print heatmap of answers per model')
    parser.add_argument("--version_dir", type=str, required=False, help="Restrict the elements represented by the heatmap to a particular version")
    parser.add_argument("--data_source", type=str, required=False, help="Restrict the elements represented by the heatmap to the source of the training data")
    parser.add_argument("--variant_id", type=str, required=False, help="Restrict the elements represented by the heatmap to a variant identifier")
    parser.add_argument("--batch_size", type=int, required=False, help="Restrict the elements represented by the heatmap to a batch size")
    parser.add_argument("--question_id", type=int, required=False, help="Restrict the elements represented by the heatmap to a particular question")

    return parser.parse_args()


if __name__ == "__main__":
    config = Configuration()
    args = setup_args()

    # base path to all models
    base_dir = config.get_model_root_path()

    output_file = config.get_path_for_datafile("model_descriptions_analysis.txt")

    data_frame = {
        "answer_ids": [],
        "model_identifier": []
    }
    with open(output_file, 'w') as file:

        print(f"base_dir: {base_dir}")
        for version_dir in os.listdir(base_dir):

            version_dir_path = os.path.join(base_dir, version_dir)
            print(f"Version dir: {version_dir_path}")
            for model_dir in os.listdir(version_dir_path):

                model_dir_path = os.path.join(version_dir_path, model_dir)
                print(f"Model: {model_dir_path}")

                description_file = os.path.join(model_dir_path, "description.json")

                if os.path.exists(description_file):    
                    # Open and read the JSON file
                    with open(description_file, 'r') as json_file:
                        try:
                            data = json.load(json_file)
                        except Exception as e:
                            log.error(f"Unable to parse file: {description_file}, error: {e}")
                            continue
                    
                    index = data['base_path'].find('_')
                    data_source = data['base_path'][index:]
                    model_id = _generate_model_id(version_dir, data['question_id'], data_source)

                    # Write the path and contents of the JSON file to the output file
                    file.write(f"Path: {model_dir_path}\n")
                    file.write(f"Question Id: {data['question_id']} batch size: {data['batch_size']} variant: {data['batch_variant_id']}\n")
                    file.write(f"Base Path: {data['base_path']}\n")
                    file.write(f"Id: {model_id}\n")
                    file.write(f"\nAnswers: \n")

                    add_model_to_heatmap = True
                    add_model_to_heatmap &= (not args.version_dir or args.version_dir == version_dir)
                    add_model_to_heatmap &= (not args.question_id or args.question_id == data['question_id'])
                    add_model_to_heatmap &= (not args.data_source or args.data_source == data_source)
                    add_model_to_heatmap &= (not args.variant_id or args.variant_id == data['batch_variant_id'])
                    add_model_to_heatmap &= (not args.batch_size or args.batch_size == data['batch_size'])

                    if add_model_to_heatmap:
                        for answer in data['answer_batch']:
                            data_frame["answer_ids"].append(answer['answer_id'])
                            data_frame["model_identifier"].append(model_id)

                    for i in range(0, len(data['answer_batch']), 4):
                        group = data['answer_batch'][i:i+4]

                        # Write ids side by side
                        for answer in group:
                            file.write(f"id: {answer['answer_id']}\t")
                        file.write("\n")
                        
                        # Write score_1 side by side
                        for answer in group:
                            file.write(f"score_1: {answer['score_1']}{_tabs_for_alignment(answer['answer_id'])}")
                        file.write("\n")

                        # Write score_2 side by side
                        for answer in group:
                            file.write(f"score_2: {answer['score_2']}{_tabs_for_alignment(answer['answer_id'])}")
                        file.write("\n")

                        file.write("\n")

                file.write(f"###############################################\n\n")
    

    # Heat map of answers per model
    df = pd.DataFrame({'answer_id': data_frame["answer_ids"], 'model_identifier': data_frame["model_identifier"]})

    df['model_num'] = df['model_identifier'].str.extract('_(\d+)_[A-F]_[a-z]*$').astype(int)
    asc_df = df.sort_values('model_num', ascending=True)
    asc_df = asc_df.drop('model_num', axis=1)

    asc_df['model_identifier'] = pd.Categorical(asc_df['model_identifier'], categories=asc_df['model_identifier'].unique(), ordered=True)

    print(asc_df)

    plt.figure(figsize=(100, 100))
    plt.title("Answer distribution across models")

    cross_tab = pd.crosstab(asc_df['answer_id'], asc_df['model_identifier'])
    sns.heatmap(cross_tab, cmap="PuBuGn", cbar=False)
    plt.show()
