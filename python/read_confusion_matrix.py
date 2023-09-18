import json
from config import Configuration

# TODO: outdated
def _print_cm(config, training_data_source, test_data_source):
	with open(config.get_path_for_datafile(f"{training_data_source}_{test_data_source}_confusion_matrices.json"), "r") as file:
		data = json.load(file)
	
	with open(config.get_path_for_datafile(f"{training_data_source}_{test_data_source}_confusion_matrices.txt"), 'w') as file:
		for key, entry in data.items():
			file.write(f"Key: {key}\n")
			file.write(f"Path: {entry['path']}\n")
			for row in entry["cm_matrix"]:
				file.write(",".join(map(str, row)) + "\n")
		file.write("###################################################\n\n")

if __name__ == "__main__":
	config = Configuration()
	_print_cm(config, 'ai', 'ai-training')
	_print_cm(config, 'ai', 'ai-rating')
	_print_cm(config, 'ai', 'man-rating')
	_print_cm(config, 'man', 'man-training')
	_print_cm(config, 'man', 'man-rating')
	_print_cm(config, 'man', 'ai-rating')