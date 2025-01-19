# Grant HRCS Tagging Model
Machine learning classifier model for tagging research grants with HRCS Health Category and Research Activity Code tags based on title and grant abstract.

### Data
Aknowledgement, the data used for this project was compiled as part of the UK Health Research Analysis studies: UK Health Research Analysis 2022 (UK Clinical Research Collaboration , 2023) https://hrcsonline.net/reports/analysis-reports/uk-health-research-analysis-2022/.

To download data and compile into a cleaned training dataset, use the below command:
```shell
make build_dataset
```
- This command downloads the tagged Excel data from from https://hrcsonline.net/.
- Then calls a Python script that compiles these datasets into single cleaned parquet files.
- Each parquet file represents a tag type with one file for RAC division, RAC group and Health Category.
- Each row represents a grant and tag combination, there can be multiple rows/tags per grant.
