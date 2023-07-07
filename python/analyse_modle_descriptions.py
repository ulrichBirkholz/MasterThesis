import os
import json
from config import Configuration
import logging as log

def _tabs_for_alignment(text, tabstop=8):
	if len(text) >= 32:
		tabs = 8
	else:
		tabs = 1
	return "\t" * tabs

if __name__ == "__main__":
	config = Configuration()

	# path to all models
	base_dir = config._get_base_model_path()

	output_file = config.get_path_for_datafile("model_descriptor_analysis.txt")

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

					# Write the path and contents of the JSON file to the output file
					file.write(f"Path: {model_dir_path}\n")
					file.write(f"Question Id: {data['question_id']} batch size: {data['batch_size']} variant: {data['batch_variant_id']}\n")
					file.write(f"Descriptor: {data['descriptor']}\n")
					file.write(f"\nAnswers: \n")

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
