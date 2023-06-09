import json
import hashlib
from dataclasses import dataclass
from typing import List

@dataclass
class BatchSize:
    size: int
    ids: List[str]
	

class Configuration():
	def __init__(self):
		# Load the configuration file
		with open('config.json') as config_file:
			self.config = json.load(config_file)

	def get_config(self):
		return self.config
	
	def get_path_for_datafile(self, name):
		return f"{self.config['data_path']}/{name}"

	def _get_base_model_path(self):
		return self.config['model_path']

	def _get_path_for_model(self, suffix):
		return f"{self.config['model_path']}/{self.config[suffix]}"

	# tsv files
	def get_questions_path(self):
		return self.get_path_for_datafile(self.config["questions"])
	
	def get_ai_answers_path(self, id=None):
		if id is None or len(id) == 0:
			replacement = ""
		else:
			replacement = f"_{id}"
		return self.get_path_for_datafile(self.config["ai_answers"].replace('#', replacement))
	
	def get_ai_unrated_answer_path(self):
		return self.get_path_for_datafile(self.config["ai_unrated_answer_path"])

	def get_man_answers_path(self):
		return self.get_path_for_datafile(self.config["man_answers"])

	def get_key_elements_path(self):
		return self.get_path_for_datafile(self.config["key_elements"])
	
	def get_ai_answers_to_rate_path(self):
		return self.get_path_for_datafile(self.config["ai_answers_to_rate"])
	
	def get_ai_answers_for_training_path(self):
		return self.get_path_for_datafile(self.config["ai_answers_for_training"])
	
	def get_man_answers_to_rate_path(self):
		return self.get_path_for_datafile(self.config["man_answers_to_rate"])

	def get_man_answers_for_training_path(self):
		return self.get_path_for_datafile(self.config["man_answers_for_training"])

	def get_rated_answers_path(self, model_descriptor, answer_descriptor, batch_size, id):
		suffix = f"{model_descriptor}_model_{answer_descriptor}_answers_{batch_size}_{id}"
		file_name = self.config['rated_answers'].replace('#', suffix)
		return self.get_path_for_datafile(file_name)
	
	def get_model_path_descriptor(self, question:str, batch_size:int, batch_id:str, descriptor:str):
		return f"{question}_{batch_size}_{batch_id}_{descriptor}"

	# AI Model
	def get_trained_bert_model_path(self, question:str, batch_size:int, batch_id:str, descriptor:str):
		path_suffix = hashlib.md5(self.get_model_path_descriptor(question, batch_size, batch_id, descriptor).encode()).hexdigest()
		return f"{self._get_path_for_model('trained_bert_version')}/{path_suffix}/"

	def get_trained_xg_boost_model_path(self, question:str, batch_size:int, batch_id:str, descriptor:str):
		path_suffix = hashlib.md5(self.get_model_path_descriptor(question, batch_size, batch_id, descriptor).encode()).hexdigest()
		return f"{self._get_path_for_model('trained_xg_boost_version')}/{path_suffix}/"

	def get_alpaca_7B_model_and_path(self):
		name = "alpaca_7B"
		return self._get_path_for_model(name), self.config[name]

	def get_batch_sizes(self) -> List[BatchSize]:
		return [BatchSize(batch_size["size"], batch_size["ids"]) for batch_size in self.config["batch_sizes"]]