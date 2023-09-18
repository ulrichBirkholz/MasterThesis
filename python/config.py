import json
import hashlib
from dataclasses import dataclass
from typing import List


@dataclass
class Batch:
    """ Represents the size and associated variant Ids of a Batch of samples.

    This class provides a structured way to capture both the number of entities
    in the batch and the unique identifiers for each variant of the batch

    Attributes:
        size (int): The number of samples contained in the batch.
        ids (List[str]): A list of unique identifiers for each variant of the batch
    """
    size: int
    ids: List[str]
    

class Configuration():
    """ Provides methods to access and manipulate the project's configuration stored in 'config.json'.
    
    This class simplifies access to the projects configuration. It also ensures consistency in the way
    configurations are used and makes it easier to maintain and update it

    Attributes:
        config (dict): A dictionary loaded from the 'config.json' file containing project configurations.
    """


    def __init__(self):
        """Initializes the Configuration object by loading the 'config.json' file."""
        with open('config.json') as config_file:
            self.config = json.load(config_file)


    def get_path_for_datafile(self, name:str) -> str:
        """ Returns the absolute path for a specific data file.

        Args:
            name (str): Name of the data file.

        Returns:
            str: Absolute path to the data file.
        """
        return f"{self.config['data_path']}/{name}"


    @staticmethod
    def _replace_str(config:str, replacement:str = "") -> str:
        """ Utility function to replace a static placeholder '#' in a configuration string

        Args:
            config (str): The configuration string with placeholder
            replacement (str, optional): String to replace the placeholder. Defaults to an empty string

        Returns:
            str: Modified configuration string with placeholder replaced
        """
        return config.replace('#', f"_{replacement}" if replacement else "")


    def get_distribution_path(self) -> str:
        """ Retrieve the path for the distribution of categories analysis results

        Returns:
            str: Path to the file with distribution of categories analysis results
        """
        return self.get_path_for_datafile(self.config["distribution"])


    # tsv files
    def get_questions_path(self) -> str:
        """ Retrieve the path for the file containing the questions

            Returns:
                str: Path to the file containing the questions
        """
        return self.get_path_for_datafile(self.config["questions"])
    

    def get_samples_path(self, id) -> str:
        """ Retrieve the path for the file containing the answers

        Returns:
            str: Path to the file containing the answers
        """
        filename = Configuration._replace_str(self.config["samples"], id)
        return self.get_path_for_datafile(filename)
    

    def get_unrated_samples_path(self) -> str:
        """ Retrieve the path for the file containing the unrated answers or samples

        Returns:
            str: Path to the file containing the unrated answers or samples
        """
        return self.get_path_for_datafile(self.config["unrated_samples_path"])


    def get_all_expert_samples_src_path(self) -> str:
        """ Retrieve the path for the file containing samples or answers annotated by experts

        Returns:
            str: Path to the file containing samples or answers annotated by experts
        """
        return self.get_path_for_datafile(self.config["expert_samples_src"])


    def get_key_elements_path(self) -> str:
        """ Retrieve the path for the file containing all key elements

        Returns:
            str: Path to the file containing all key elements
        """
        return self.get_path_for_datafile(self.config["key_elements"])
    

    def get_samples_for_testing_path(self, model_version:str) -> str:
        """ Retrieve the path for the file containing samples reserved for testing, based on a specified source
        
        Args:
            model_version (str): Source identifier for the samples (e.g davinci, turbo, gpt4, experts)
            
        Returns:
            str: Path to the samples file
        """
        filename = Configuration._replace_str(self.config["samples_for_testing"], model_version)
        return self.get_path_for_datafile(filename)
    

    def get_samples_for_training_path(self, model_version:str):
        """ Retrieve the path for the file containing samples reserved for training, based on a specified source
        
        Args:
            model_version (str): Source identifier for the samples (e.g davinci, turbo, gpt4, experts)
            
        Returns:
            str: Path to the samples file
        """
        filename = Configuration._replace_str(self.config["samples_for_training"], model_version)
        return self.get_path_for_datafile(filename)


    def get_test_results_path(self, training_data_source:str, test_data_source:str, batch_size:int, batch_id:str) -> str:
        """ Retrieve the path for the file containing annotated answers created by a model trained with specific data
        Args:
            training_data_source (str): Source of the data used to train the model (e.g., davinci, turbo, gpt4, experts)
            test_data_source (str): Source of the data the model was tested with (e.g., davinci, turbo, gpt4, experts)
            batch_size (int): Number of samples used
            batch_id (str): Variant identifier

        Returns:
            str: Path to the file containing the results
        """
        suffix = f"{training_data_source}_model_rated_{test_data_source}_samples_{batch_size}_{batch_id}"
        filename = Configuration._replace_str(self.config["test_results"], suffix)
        return self.get_path_for_datafile(filename)
    

    def get_relative_model_path(self, question:str, batch_size:int, batch_id:str, training_data_source:str) -> str:
        """ Generate a relative path for a model based on its training information

        Args:
            question (str): Question the model addresses
            batch_size (int): The number of samples the model was trained with
            batch_id (str): Variant identifier
            training_data_source (str): Source of the data used to train the model (e.g., davinci, turbo, gpt4, experts)

        Returns:
            str: The path to the model relative to the base folder of all models
        """
        return f"{question}_{batch_size}_{batch_id}_{training_data_source}"


    def get_lr_calculations_path(self) -> str:
        """ Retrieve the path for the lexical diversity analysis results

        Returns:
            str: Path to the file with lexical diversity analysis results
        """
        return self.get_path_for_datafile(self.config["lr_calculations"])


    def get_qwk_result_path(self) -> str:
        """ Retrieve the path for the file containing the calculated Quadratic Weighted Kappa (QWK)

        Returns:
            str: Full path to the file containing the calculated QWK
        """
        return self.get_path_for_datafile(self.config['qwk_path'])


    # AI Model
    def get_model_root_path(self) -> str:
        """ Retrieve the base directory path for all models

        Returns:
            str: Base path for all model directories
        """
        return self.config['model_path']


    def _get_path_for_model(self, model_version:str) -> str:
        """ Retrieve the root directory for a specific model type and version
        
        Args:
            model_version (str): Model type and version identifier
        
        Returns:
            str: Base directory path for the specified model type and version
        """
        return f"{self.get_model_root_path()}/{self.config[model_version]}"
    

    def get_trained_bert_model_path(self, question:str, batch_size:int, batch_id:str, training_data_source:str) -> str:
        """ Retrieve the path for a specific BERT model based on its training data

        Args:
            question (str): Question the model addresses
            batch_size (int): The number of samples the model was trained with
            batch_id (str): Variant identifier
            training_data_source (str): Source of the data used to train the model (e.g., davinci, turbo, gpt4, experts)

        Returns:
            str: Path for the BERT model matching the specified criteria
        """
        path_suffix = hashlib.md5(self.get_relative_model_path(question, batch_size, batch_id, training_data_source).encode()).hexdigest()
        return f"{self._get_path_for_model('trained_bert_version')}/{path_suffix}/"


    def get_trained_xg_boost_model_path(self, question:str, batch_size:int, batch_id:str, training_data_source:str) -> str:
        """ Retrieve the path for a specific XG-Boost model based on its training data

        Args:
            question (str): Question the model addresses
            batch_size (int): The number of samples the model was trained with
            batch_id (str): Variant identifier
            training_data_source (str): Source of the data used to train the model (e.g., davinci, turbo, gpt4, experts)

        Returns:
            str: Path for the XG-Boost model matching the specified criteria
        """
        path_suffix = hashlib.md5(self.get_relative_model_path(question, batch_size, batch_id, training_data_source).encode()).hexdigest()
        return f"{self._get_path_for_model('trained_xg_boost_version')}/{path_suffix}/"


    # Other Configurations
    def get_batches(self) -> List[Batch]:
        """ Retrieve all configured batches

        Returns:
            List[Batch]: List of all configured batches
        """
        return [Batch(batch["size"], batch["ids"]) for batch in self.config["batches"]]