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

	def get_config(self) -> str:
		return self.config
	
	def get_path_for_datafile(self, name:str) -> str:
		return f"{self.config['data_path']}/{name}"

	def _get_base_model_path(self) -> str:
		return self.config['model_path']

	def _get_path_for_model(self, suffix:str) -> str:
		return f"{self.config['model_path']}/{self.config[suffix]}"

	# We always use '#' as placeholder
	@staticmethod
	def _replace_str(config:str, replacement:str = "") -> str:
		return config.replace('#', f"_{replacement}" if replacement else "")

	def get_ttr_calculations_path(self) -> str:
		return self.get_path_for_datafile(self.config["ttr_calculations"])
	
	def get_distribution_path(self) -> str:
		return self.get_path_for_datafile(self.config["distribution"])

	# tsv files
	def get_questions_path(self) -> str:
		return self.get_path_for_datafile(self.config["questions"])
	
	def get_samples_path(self, id) -> str:
		filename = Configuration._replace_str(self.config["samples"], id)
		return self.get_path_for_datafile(filename)
	
	def get_unrated_samples_path(self) -> str:
		return self.get_path_for_datafile(self.config["ai_unrated_answer_path"])

	def get_all_expert_samples_src_path(self) -> str:
		return self.get_path_for_datafile(self.config["expert_samples_src"])

	def get_key_elements_path(self) -> str:
		return self.get_path_for_datafile(self.config["key_elements"])
	
	def get_samples_for_testing_path(self, model_version:str) -> str:
		filename = Configuration._replace_str(self.config["samples_for_testing"], model_version)
		return self.get_path_for_datafile(filename)
	
	def get_samples_for_training_path(self, model_version:str):
		filename = Configuration._replace_str(self.config["samples_for_training"], model_version)
		return self.get_path_for_datafile(filename)

	def get_test_results_path(self, training_data_source:str, test_data_source:str, batch_size:int, id:str) -> str:
		suffix = f"{training_data_source}_model_rated_{test_data_source}_samples_{batch_size}_{id}"
		filename = Configuration._replace_str(self.config["test_results"], suffix)
		return self.get_path_for_datafile(filename)
	
	def get_model_base_path(self, question:str, batch_size:int, batch_id:str, training_data_source:str) -> str:
		return f"{question}_{batch_size}_{batch_id}_{training_data_source}"

	# AI Model
	def get_trained_bert_model_path(self, question:str, batch_size:int, batch_id:str, descriptor:str, chat_gpt_model:str) -> str:
		path_suffix = hashlib.md5(self.get_model_base_path(question, batch_size, batch_id, descriptor, chat_gpt_model).encode()).hexdigest()
		return f"{self._get_path_for_model('trained_bert_version')}/{path_suffix}/"

	def get_trained_xg_boost_model_path(self, question:str, batch_size:int, batch_id:str, descriptor:str, chat_gpt_model:str) -> str:
		path_suffix = hashlib.md5(self.get_model_base_path(question, batch_size, batch_id, descriptor, chat_gpt_model).encode()).hexdigest()
		return f"{self._get_path_for_model('trained_xg_boost_version')}/{path_suffix}/"

	def get_alpaca_7B_model_and_path(self) -> str:
		name = "alpaca_7B"
		return self._get_path_for_model(name), self.config[name]

	def get_batch_sizes(self) -> List[BatchSize]:
		return [BatchSize(batch_size["size"], batch_size["ids"]) for batch_size in self.config["batch_sizes"]]