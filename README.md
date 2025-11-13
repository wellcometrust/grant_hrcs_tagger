# Grant HRCS Tagging Model
Machine learning classifier model for tagging research grants with HRCS Health Category and Research Activity Code tags based on title and grant abstract.

Developed by the Machine Learning team, within Data & Digital at [the Wellcome Trust](https://wellcome.org/).

### Data
Acknowledgment, for training and evaluation of our model we used the following datasets:
* UK Health Research Analysis 2014, 2018 and 2022 reports from HRCS online: https://hrcsonline.net/. E.g. the 2022 report data can be found here: https://hrcsonline.net/reports/analysis-reports/uk-health-research-analysis-2022/.
* NIHR tagged awards dataset: https://nihr.opendatasoft.com/api/explore/v2.1/catalog/datasets/nihr-summary-view/exports/parquet?lang=en&timezone=Europe%2FLondon

## To use the latest model for tagging
[instructions to be added on how to download and use the latest trained model via Huggingface]

## To set up the project for training and development

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

# Get help and see all available commands
make help
```


