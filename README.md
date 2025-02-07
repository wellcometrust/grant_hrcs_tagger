# Grant HRCS Tagging Model
Machine learning classifier model for tagging research grants with HRCS Health Category and Research Activity Code tags based on title and grant abstract.

Dveloped by the Machine Learning team, within Data & Digital at the Wellcome Trust.

### Data
Aknowledgement, the data used for this project was compiled as part of the UK Health Research Analysis studies: UK Health Research Analysis 2022 (UK Clinical Research Collaboration , 2023) https://hrcsonline.net/reports/analysis-reports/uk-health-research-analysis-2022/.

Aknowledgement, the data used for this project was compiled as part of the UK Health Research Analysis studies: UK Health Research Analysis 2022 (UK Clinical Research Collaboration , 2023) https://hrcsonline.net/reports/analysis-reports/uk-health-research-analysis-2022/.

## Set up

### 1. Environment set up

Start with setting up the virtual environment for this project. Make sure you have conda installed as we will use it as an environment manager. [If conda is not installed, installing miniconda is a good starting point](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)

:green_apple: On Mac M1 `conda env create -f environment_mac.yml` 
:penguin: On Linux `conda env create -f environment.yml` 

The environment can be activates with `conda activate hrcs_tagger`

### 2. Downloading the dataset

To Download the UK Health Research Analysis data used for training, run: 

```shell
make build_dataset
```
- This command downloads the tagged Excel data from from https://hrcsonline.net/.
- Then calls a Python script that compiles these datasets into single cleaned parquet files.
- Each parquet file represents a tag type with one file for RAC division, RAC group and Health Category.
- Each row represents a grant and tag combination, there can be multiple rows/tags per grant.

This make command assumes `wget` is installed, which on a Mac you will have to install first, `brew install wget`.
