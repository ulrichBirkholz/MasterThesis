import json
from config import Configuration
import os


def _cleanup(config:Configuration) -> None:
    """ Ensures that none of the files, created by this module, already exist

    Args:
        config (Configuration): Allows access to the projects central configuration
    """
    results_root_path = config.get_results_root_path()
    for data_file in os.listdir(results_root_path):
        if data_file.endswith("_confusion_matrices.txt"):
            full_path = os.path.join(results_root_path, data_file)
            os.remove(full_path)


if __name__ == "__main__":
    config = Configuration()
    results_root_path = config.get_results_root_path()
    _cleanup(config)

    for data_file in os.listdir(results_root_path):
        if data_file.endswith("_confusion_matrices.json"):
            text_file_name = data_file.replace(".json", ".txt")

            with open(config.get_path_for_results_file(data_file), "r") as file:
                data = json.load(file)
            
            with open(config.get_path_for_results_file(text_file_name), 'w') as file:
                for key, entry in data.items():
                    file.write(f"Key: {key}\n")
                    file.write(f"Path: {entry['path']}\n")
                    for row in entry["cm_matrix"]:
                        file.write(",".join(map(str, row)) + "\n")
                file.write("###################################################\n\n")