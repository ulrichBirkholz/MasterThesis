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

### For Expert Samples

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

TODO: update score_types.json based on distribution result

## Train Models

Use the `train_model` module to train the BERT and XG-Boost models on the generated samples.

The configuration key `model_path` points to the primary directory where all model versions are stored. Within this main directory, each model version has a dedicated sub-folder. The naming convention for these sub-folders is based on an MD5 hash derived from a combination of the question, batch_size, batch_id, and training_data_source.

TODO: mention descriptor.json

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