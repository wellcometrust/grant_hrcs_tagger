# Grant HRCS Tagging Model
Machine learning classifier model for tagging research grants with HRCS Health Category and Research Activity Code tags based on title and grant abstract.


Aknowledgement, the data used for this project was compiled as part of the UK Health Research Analysis studies: UK Health Research Analysis 2022 (UK Clinical Research Collaboration , 2023) https://hrcsonline.net/reports/analysis-reports/uk-health-research-analysis-2022/.

## Set up

### 1. Environment set up

Start with setting up the virtual environment for this project. Make sure you have conda installed as we will use it as an environment manager. [If conda is not installed, installing miniconda is a good starting point](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)

:green_apple: On Mac M1 `conda env create -f environment_mac.yml` 
:penguin: On Linux `conda env create -f environment.yml` 

The environment can be activates with `conda activate hrcs_tagger`
