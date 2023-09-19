# Master's Thesis in Computational Linguistics


## Structure

```
.
├── MasterThesis.pdf    # Compiled version of the thesis
├── thesis_tex          # LaTeX source files
├── python              # Python code
└── data                # Produced samples, diagrams, and analyses
```

## Prerequisites

- Open AI API key
- A working [conda](https://docs.conda.io/) installation

## Installation

1. Create a [conda](https://docs.conda.io/) environment named `master_thesis`:
   ```bash
   conda env create -f environment.yaml
   ```

2. Activate the environment:
   ```bash
   conda activate master_thesis
   ```

## Analyse System Resources  TODO: check wording

The module `analyse_gpu` allows to analyze the systems resources and verify that torch is able to use the GPU.
The file outputs the number of GPU's and their respective memory.

## Generate Unannotated Samples

Execute the `generate_samples` module to produce:
- `unrated_samples_davinci.tsv`: Contains answers produced by `text-davinci-003`
- `unrated_samples_gpt4.tsv`: Contains answers produced by `gpt4`

Use the `calculate_lexical_richness` module to measure the lexical richness of all files. Results are saved in `lexical_richness_calculations.tsv`.

## Annotate Samples

Execute the `annotate_samples` module to annotate the generated answers. It will produce annotations using various models, and save them in their respective `.tsv` files.

### For Unrated Samples

1. **Source File**: `unrated_samples_davinci.tsv`
    - **Model `text-davinci-003`**: Annotations saved in `samples_davinci.tsv`
    - **Model `gpt-3.5-turbo`**: Annotations saved in `samples_turbo.tsv`

2. **Source File**: `unrated_samples_gpt4.tsv`
    - **Model `gpt4`**: Annotations saved in `samples_gpt4.tsv`

### For Expert Samples TODO: check wording

The module `pick_expert_samples` was used to extract the answers assigned to Essey Set 5 and 6 from the file containing the samples for all Essey Sets `train.tsv`.

**Source File**: `samples_experts.tsv`

Annotations will be produced using multiple models:
- **Model `text-davinci-003`**: Annotations saved in `samples_davinci_rating_expert_data.tsv`
- **Model `gpt-3.5-turbo`**: Annotations saved in `samples_turbo_rating_expert_data.tsv`
- **Model `gpt4`**: Annotations saved in `samples_gpt4_rating_expert_data.tsv`


## Split Sample Sets

The `pick_random_samples` module splits the sample sets into training and testing datasets. The distribution of categories within the sample sets is saved in `distribution.txt`.

Utilize the module to divide the sample sets into two distinct groups:
1. **Training Set**: Used for training the models.
2. **Test Set**: Employed for testing the models.

### Sample Set Details:

- **ChatGPT Samples**:
    - Contains 3200 samples per Essay Set for training.
  
- **Human Expert Samples**:
    - Contains only 1600 samples for training, being a more limited set.

The module also computes the category distribution for each sample set and records this information in the `distribution.txt` file.

TODO: check wording
This file will show which score type (either. 1 for score_1 or 2 for score_2) for which question and data source should be used to get the best results.

Enter the score types to be used into the score_types.json:
```json
{
	"davinci": { // Score types for answers annotated by text-davinci-003
		"5":1, // Score type for Essey Set 5, in this case score_1
		"6":1 // Score type for Essey Set 6, in this case score_1
	},
	"turbo": { // Score types for answers annotated by gpt-3.5-turbo
		"5":2, // Score type for Essey Set 5, in this case score_2
		"6":2
	},
	"experts": { // Score types for answers annotated by human experts
		"5":2,
		"6":1
	},
	"gpt4": { // Score types for answers annotated by gpt4
		"5":1,
		"6":2
	}
}
```

## Train Models

Use the `train_model` module to train the BERT and XG-Boost models on the generated samples.

The configuration key `model_path` points to the primary directory where all model versions are stored. Within this main directory, each model version has a dedicated sub-folder. The naming convention for these sub-folders is based on an MD5 hash derived from a combination of the question, batch_size, batch_id, and training_data_source.

The file `descriptor.json` provides relevant information about the respective model TODO: check wording
```json
{
	"answer_batch": [ // All answers the model was trained with
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
	"question_id": "5", // The Essey Set the answers are associated with
	"question": "A scientific question", // The plain text of the Essey Sets Prompt
	"batch_size": 50, // The number of answers the model was trained with
	"batch_variant_id": "A", // The Id of the Variant the model represents
	"base_path": "A scientific question_50_A_turbo", // The information used to crated the models folder name, which is an MD5 hash of this String
	"epochs": 10, // The number of epoches the model was trained with
	"existing_batches": 2 // The number of selected batches from the respective set of samples
}
```
TODO: mention the module analyse_model_descriptions
## Test Models

The `test_model` module evaluates the trained models on the test samples, saving the results in separate `.tsv` files. It also produces a confusion matrix for each essay set. 

The confusion matrixes will be saved at {training_data_source}_{test_data_source}_{question.question_id}_confusion_matrices.json

TODO: read_confusion_matrix.py

## Evaluate Results

Execute the `calculate_kappa` module to evaluate the results. This module computes various QWK combinations and saves the results in `qwk.tsv`. It also produces diagrams displaying these results.

## Project Configuration

Below is the project's JSON configuration file detailing various settings:
```json
{
    // Relative path to the data folder (contains TSV files, diagrams, etc.)
	"data_path": "../data",

    // Relative path to the models folder (stores trained models)
	"model_path": "../models",

    // BERT model's internal version
	"trained_bert_version": "bert_v1",

    // XG-Boost model's internal version
	"trained_xg_boost_version": "xgb_v1",

    // Name of the TSV file containing the Questions
	"questions": "questions.tsv",

    // Template for sample answers TSV filenames. '#' is replaced with the data source (e.g., davinci, gpt4, turbo, experts)
	"samples": "samples#.tsv",

    // Template for training samples. Replace '#' as above
	"samples_for_training": "samples_for_training#.tsv",

    // Template for testing samples. Replace '#' as above
	"samples_for_testing": "samples_for_testing#.tsv",

    // Template for unannotated answers. Replace '#' as above.
	"unrated_samples_path": "unrated_samples#.tsv",

    // TSV file containing original answers from all ASAP Essay Sets
	"expert_samples_src": "train.tsv",

    // Name tsv file containing the key elements for ASAP Essey Sets 5 and 6
	"key_elements": "key_elements.tsv",

    // Template for test results file. '#' is replaced based on model platform, data source, sample count, and variant Id.
	"test_results": "test_results#.tsv",

    // Stores results of lexical diversity metric calculations
	"lr_calculations": "lexical_richness_calculations.tsv",

    // File with computed QWK values
	"qwk_path": "qwk.tsv",

    // File detailing category distribution across datasets
	"distribution": "distribution.txt",

    // Configuration for batches in terms of size and variant Ids
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

## Review all possible prompts or messages TODO: check wording

The module `print_gpt_instructions` can be used to print all prompts and messages used to produce the samples.
