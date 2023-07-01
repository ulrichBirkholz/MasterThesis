import json
from config import Configuration

def _print_cm(config, model_descriptor, answer_descriptor):
	with open(config.get_path_for_datafile(f"{model_descriptor}_{answer_descriptor}_confusion_matrices.json"), "r") as file:
		data = json.load(file)
	
	with open(config.get_path_for_datafile(f"{model_descriptor}_{answer_descriptor}_confusion_matrices.txt"), 'w') as file:
		for key, cm in data.items():
			file.write(key + "\n")
			for row in cm:
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