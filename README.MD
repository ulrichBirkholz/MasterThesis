# Master's Thesis in Computational Linguistics


## Repository Structure

```
.
├── MasterThesis.pdf    # Compiled PDF of the thesis
├── thesis_tex          # LaTeX source files for the thesis
├── python              # Python code utilized in the research
└── data                # Generated samples, diagrams, and analysis results
	├── chat_gpt_cm		# ChatGPT's confusion matrices and benchmarks
	├── lr				# Results and diagrams of the lexical richness calculations
	├── test_results	# Test results of the trained BERT and XG-Boost models, including confusion matrices
	├── performance		# Calculations and Visual Representations for Various Performance Metrics, e.g., F1 Score
	└── qwk				# Results and diagrams of the QWK calculations
```

## Required Dependencies

- OpenAI API key
- A functional [conda](https://docs.conda.io/) installation

## Setup Instructions

1. Set up a [conda](https://docs.conda.io/) environment named `master_thesis` using the provided configuration:
   ```bash
   conda env create -f environment.yaml
   ```

2. Once created, activate the environment:
   ```bash
   conda activate master_thesis
   ```

## System Resources Analysis 

The `analyse_gpu` module enables you to inspect system resources. Specifically, it checks if `torch` can access and utilize the GPU. The output provides a count of available GPUs along with their respective memory capacities.

## Generating Unannotated Samples

Run the `generate_samples` module to create the following files:
- `unrated_samples_davinci.tsv`: This file includes responses generated by `text-davinci-003`.
- `unrated_samples_gpt4.tsv`: This file captures answers produced using `gpt4`.

For assessing the lexical richness of the generated content, invoke the `calculate_lexical_richness` module. The computed results are stored in the `lexical_richness_calculations.tsv` file, and display the metrics as diagrams.

## Sample Annotation

Run the `annotate_samples` module to process and annotate the generated answers. The module will use multiple models for annotations and store the results in corresponding `.tsv` files. To calculate the corresponding confusion matrices and benchmarks execute the module `results_to_confustion_matrix`.

Note: The model `gpt-3.5-turbo` occasionally struggled with accurate categorization. To address this, we introduced the `re_rate_missing_answers` module. It ensures that every answer crafted by human experts is rated and included in the dataset produced by this specific model.

### Annotations for Unrated Samples:

1. **Source File**: `unrated_samples_davinci.tsv`
    - **Model `text-davinci-003`**: Annotations stored in `samples_davinci.tsv`
    - **Model `gpt-3.5-turbo`**: Annotations stored in `samples_turbo.tsv`

2. **Source File**: `unrated_samples_gpt4.tsv`
    - **Model `gpt4`**: Annotations stored in `samples_gpt4.tsv`

### Annotations for Expert Samples

Use the `pick_expert_samples` module to select answers designated for Essay Sets 5 and 6 from the comprehensive sample file: `train.tsv`.

**Source File**: `samples_experts.tsv`

Annotations will be generated by several models:
- **Model `text-davinci-003`**: Annotations stored in `samples_davinci_rating_expert_data.tsv`
- **Model `gpt-3.5-turbo`**: Annotations stored in `samples_turbo_rating_expert_data.tsv`
- **Model `gpt4`**: Annotations stored in `samples_gpt4_rating_expert_data.tsv`


## Sample Set Segregation

The `pick_random_samples` module facilitates the division of sample sets into two discrete partitions: the training and testing datasets. The breakdown and categorization within these sample sets are documented in the `distribution.txt` file.

By employing this module, samples are bifurcated into:
1. **Training Set**: Dedicated for model training.
2. **Test Set**: Reserved for model validation and testing.

### Breakdown of Sample Sets:

- **ChatGPT Samples**:
    - Encompasses 3200 samples for each Essay Set designated for training.
  
- **Human Expert Samples**:
    - Comprises a more exclusive collection of 1600 samples for each Essay Set for training.

Additionally, this module evaluates and delineates the category distribution for every sample set, recording these metrics in the `distribution.txt` file.

The content of this file offers insights into the optimal score type (either 1 for `score_1` or 2 for `score_2`) corresponding to specific questions and data sources that yield the best results.

To apply these recommendations, input the desired score types into the `score_types.json` file:
```json
{
	"davinci": { // Score categorization for answers annotated by text-davinci-003
		"5":1, // Preferred score type for Essay Set 5, in this instance, score_1
		"6":1 // Preferred score type for Essay Set 6, in this instance, score_1
	},
	"turbo": { // Score categorization for answers annotated by gpt-3.5-turbo
		"5":2, // Preferred score type for Essay Set 5, in this instance, score_2
		"6":2
	},
	"experts": { // Score categorization for answers evaluated by human experts
		"5":2,
		"6":1
	},
	"gpt4": { // Score categorization for answers annotated by gpt4
		"5":1,
		"6":2
	}
}
```

## Model Training

Leverage the `train_model` module to train both the BERT and XG-Boost models using the generated samples.

The configuration parameter `model_path` specifies the root directory where all model iterations are archived. Each iteration of the model is neatly organized into its unique sub-directory within this primary location. These sub-directory names are constructed using a blend of the question, batch_size, batch_id, and training_data_source.

For a comprehensive understanding of each model version, refer to the `descriptor.json` file, which encapsulates details about the corresponding model:
```json
{
	"answer_batch": [ // Ensemble of all answers the model was refined with
		{
			"answer": "Some answer",
			"answer_id": "dde3c5865822627d6f7d2577f1526b0f",
			"score_1": "2",
			"score_2": "2"
		}
		...
		{
			"answer": "Some other answer",
			"answer_id": "f7e4395ec7da7af3bb6625b3001dd167",
			"score_1": "3",
			"score_2": "2"
		}
	],
	"question_id": "5", /// Associated Essay Set
	"question": "A scientific question", // The plain text of the Essey Set's prompt
	"batch_size": 50, // The number of answers used for training the model
	"batch_variant_id": "A", // Identifier for the model's variant
	"base_path": "5_50_A_turbo", // The folder name for the model
	"epochs": 10, // Total training cycles the model underwent
	"existing_batches": 2 // Count of curated batches from the associated sample set
}
```
The module `analyse_model_descriptions` provides an allocated description of all descriptor.json files. It stores the results in the file `model_descriptions_analysis.txt`. It also produces a heatmap of which answer is used to train which model.

## Model Testing

Utilize the `test_model` module to assess the performance of trained models against test samples. The evaluation outcomes are preserved in distinct `.tsv` files. Additionally, a confusion matrix for each essay set is generated to provide a clear visualization of the model's performance.

Confusion matrices are stored in the format: `{training_data_source}_{test_data_source}_{question.question_id}_confusion_matrices.json`

To further interpret these matrices, the `analyse_confusion_matrices` module has been developed. This module processes all the confusion matrices and transcribes them into text files named: `{training_data_source}_{test_data_source}_{question.question_id}_confusion_matrices.txt` and calculates all relevant performance metrics. These metrics are then stored in TSV format and visualized as diagrams.

## Results Evaluation

For a comprehensive assessment of the results, run the `calculate_kappa` module. This tool calculates diverse combinations of the Quadratic Weighted Kappa (QWK) and archives these findings in `qwk.tsv`. Additionally, it crafts visual diagrams to depict these outcomes more intuitively.

## Configuration Overview

Outlined below is the JSON structure representing the project's various configuration parameters:
```json
{
    // Relative path to the data folder (contains TSV files, diagrams, etc.)
	"data_path": "../data",

    // Relative path to the models folder (stores trained models)
	"model_path": "../models",

    // Specific internal version for the BERT model
	"trained_bert_version": "bert_v1",

    // Specific internal version for the XG-Boost model
	"trained_xg_boost_version": "xgb_v1",

    // Filename for the TSV file that contains the Questions
	"questions": "questions.tsv",

	// Filename format for sample answers. '#' is replaced with the data source names like davinci, gpt4, turbo, etc.
	"samples": "samples#.tsv",

    // Filename format for training samples. '#' follows the same replacement logic as above
	"samples_for_training": "samples_for_training#.tsv",

    // Filename format for test samples. '#' is replaced similarly
	"samples_for_testing": "samples_for_testing#.tsv",

    // Filename format for unannotated answers. Replace '#' as previously
	"unrated_samples_path": "unrated_samples#.tsv",

    // Filename of the TSV containing original responses from all ASAP Essay Sets
	"expert_samples_src": "train.tsv",

    // TSV filename listing the key elements for ASAP Essay Sets 5 and 6
	"key_elements": "key_elements.tsv",

    // Naming convention for the test results file. '#' is replaced based on the model platform, data source, sample count, and variant Id
	"test_results": "test_results#.tsv",

    // File containing results of the lexical diversity metrics
	"lr_calculations": "lexical_richness_calculations.tsv",

    // File with recorded QWK computation outcomes
	"qwk_path": "qwk.tsv",

    // File specifying the category distribution within datasets
	"distribution": "distribution.txt",

    // Batch configurations including their sizes and variant identifiers
	"batches": [
        {"size":50, "ids":["A", "B", "C", "D", "E", "F"]}, 
        {"size":100, "ids":["A", "B", "C", "D", "E", "F"]}, 
        {"size":200, "ids":["A", "B", "C", "D", "E", "F"]}, 
        {"size":400, "ids":["A", "B", "C", "D", "E", "F"]}, 
        {"size":800, "ids":["A", "B", "C", "D", "E", "F"]}, 
        {"size":1600, "ids":["A"]}, 
        {"size":3200, "ids":["A"]}]
}
```

Note: The '#' symbol within the filename templates is a placeholder, meant to be replaced by appropriate values depending on the context.

## Displaying Prompts and Messages

To view possible prompts and messages that were utilized for generating the samples, employ the `print_gpt_instructions` module.
