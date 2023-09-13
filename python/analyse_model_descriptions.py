import os
import json
from config import Configuration
import logging as log
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

def _tabs_for_alignment(text, tabstop=8):
	if len(text) >= 32:
		tabs = 8
	else:
		tabs = 1
	return "\t" * tabs

def _generate_model_id(descriptor:str, question_id:str, version:str):
    index = descriptor.find('_')
    return f"{version}_{question_id}{descriptor[index:]}"

if __name__ == "__main__":
	config = Configuration()

	# path to all models
	base_dir = config._get_base_model_path()

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
					

					model_id = _generate_model_id(data['descriptor'], data['question_id'], version_dir)

					# Write the path and contents of the JSON file to the output file
					file.write(f"Path: {model_dir_path}\n")
					file.write(f"Question Id: {data['question_id']} batch size: {data['batch_size']} variant: {data['batch_variant_id']}\n")
					file.write(f"Descriptor: {data['descriptor']}\n")
					file.write(f"ID: {model_id}\n")
					file.write(f"\nAnswers: \n")


					# -> Adjust this to modify the analysed dataset
					# TODO: parameterise??
					pattern = r'^bert_v1_5_(50|100|200|400)_[B]_ai$'

					if re.match(pattern, model_id):
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
