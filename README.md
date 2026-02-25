# Grant HRCS Tagging Model
Machine learning classifier model for tagging research grants with HRCS Health Category and Research Activity Code tags based on title and grant abstract.

Developed by the Machine Learning team, within Data & Digital at [the Wellcome Trust](https://wellcome.org/).

### Data
Acknowledgment, for training and evaluation of our model we used the following datasets:
* UK Health Research Analysis 2014, 2018 and 2022 reports from HRCS online: https://hrcsonline.net/. E.g. the 2022 report data can be found here: https://hrcsonline.net/reports/analysis-reports/uk-health-research-analysis-2022/.
* NIHR tagged awards dataset: https://nihr.opendatasoft.com/api/explore/v2.1/catalog/datasets/nihr-summary-view/exports/parquet?lang=en&timezone=Europe%2FLondon

## Project Structure

```
├── config/                 # Configuration files
│   ├── train_config.yaml   # Training hyperparameters and settings
│   └── deploy_config.yaml  # Deployment configuration
├── data/                   # Data directory
│   ├── raw/                # Raw downloaded data
│   ├── clean/              # Cleaned parquet files
│   ├── preprocessed/       # Train/test splits
│   ├── label_names/        # Label name mappings
│   └── model/              # Trained model outputs
├── src/                    # Source code
│   ├── data_processing/    # Data processing scripts
│   ├── train.py            # Model training script
│   ├── train_test_split.py # Data splitting utilities
│   ├── inference.py        # Inference functions
│   ├── deploy.py           # SageMaker deployment
│   └── metrics.py          # Evaluation metrics
├── notebooks/              # Jupyter notebooks for exploration
└── test/                   # Test suite
```

## To use the latest model for tagging
[instructions to be added on how to download and use the latest trained model via Huggingface]

## To set up the project for training and development

### Platform Requirements

This project uses a Makefile with bash commands, which run natively on **Linux** and **macOS**.

**Windows users:** We recommend using [WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/install) and following the Linux instructions. Makefiles are not supported natively on Windows without third-party tools, and the bash commands in the Makefile only run on Unix-like systems.

### 1. Install uv

First, install uv, a fast Python package manager. You can install it using:

```shell
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.sh | iex"

# Or with pip
pip install uv
```

For more installation options, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Environment set up

Set up the project environment using uv, which will read from the `pyproject.toml` file:

```shell
# Sync dependencies and create virtual environment
uv sync
```

You can run python scripts directly with uv without activating the virtual environment explicitly:

```shell
uv run a_python_script.py
```

To make things even easier, we use the following `make` commands to run common tasks.

### 4. Make commands

The project includes several `make` commands to streamline common tasks: 
**Note:** it is best to run these 

```
# Download raw datasets from HRCS online and NIHR
make download_data
```
This make command assumes `wget` is installed, which on a Mac you will have to install first, `brew install wget`.
```
# Process and clean the downloaded data into parquet files
make build_dataset

# Preprocess data for training (splits into train/test sets)
make preprocess

# Train models for different tag types:
make train_ra      # Train Research Activity model
make train_ra_top  # Train Research Activity (top level) model  

# See all available commands
make help
```

### 5. Fine-tune the model

To train your own HRCS tagging model, you'll need:

**Hardware Requirements:**
- A GPU is highly recommended for training. The code supports:
  - CUDA GPUs (NVIDIA)
  - Apple Silicon (M1/M2/M3/M4/...) via MPS

**Training Steps:**

1. **Download and preprocess data** (if not already done):
```shell
make download_data
make build_dataset  
make preprocess
```

2. **Configure training settings** by editing `config/train_config.yaml`:

```yaml
training_settings:
  category: 'RA'  # Choose: "RA" (Research Activity), "RA_top" (top-level), or "HC" (Health Category)
  model: 'answerdotai/ModernBERT-base'  # Also supports 'distilbert-base-uncased' or any model compatible with AutoModelForSequenceClassification
  learning_rate: 0.0001
  num_train_epochs: 3  # Increase for better performance (try 3-5)
  per_device_train_batch_size: 16  # Reduce if you get GPU memory errors
  class_weighting: False  # Set to True to handle label imbalance
  output_weighting: True  # Set to True for custom prediction thresholds, we found this to slightly improve performance
```

3. **Run training**:
```shell
# Train Research Activity model
make train_ra # for the low level RAC codes
make train_ra_top  # for the top level RAC codes
```

**Monitoring with Weights & Biases:**
The code integrates with [wandb](https://wandb.ai/) for experiment tracking. If you have a wandb account, training metrics will be automatically logged. If you don't have access to wandb, you can disable it by setting `report_to: none` in the train_config.yaml file.

Trained models are saved to `data/model/` and can be used for inference on new grants.

## Inference

The trained model can be loaded for inference using the functions in `src/inference.py`:

```python
from src.inference import model_fn, predict_fn

# Load model
model_dict = model_fn("data/model/")

# Predict
result = predict_fn({"inputs": "Grant title and abstract text here"}, model_dict)
```

The inference pipeline applies ranked thresholds (configurable in the training config) to convert logits to multi-label predictions.

## Deployment

Models can be deployed to AWS SageMaker. Configuration is managed in `config/deploy_config.yaml`:

```yaml
model_args:
  transformers_version: "4.49.0"
  pytorch_version: "2.6.0"
  py_version: "py312"

endpoint_args:
  instance_type: "ml.m5.xlarge"
```

See `src/deploy.py` for deployment utilities and `notebooks/deploy.ipynb` for an interactive deployment workflow.

## Testing

Run the test suite:

```shell
pytest test/
```
