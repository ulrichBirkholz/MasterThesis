import json
from config import Configuration
import os


if __name__ == "__main__":
	config = Configuration()
	data_root_path = config.get_datafile_root_path()

	# TODO check if list dir lists all files
	for data_file in os.listdir(data_root_path):
		if data_file.endswith("_confusion_matrices.json"):
			text_file_name = data_file.replace(".json", ".txt")

			with open(config.get_path_for_datafile(data_file), "r") as file:
				data = json.load(file)
			
			with open(config.get_path_for_datafile(text_file_name), 'w') as file:
				for key, entry in data.items():
					file.write(f"Key: {key}\n")
					file.write(f"Path: {entry['path']}\n")
					for row in entry["cm_matrix"]:
						file.write(",".join(map(str, row)) + "\n")
				file.write("###################################################\n\n")