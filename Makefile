# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# \
	Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

.PHONY: download_data
download_data:
	mkdir -p data
	mkdir -p data/raw
	mkdir -p data/clean
	wget -O data/raw/ukhra2022.xlsx https://hrcsonline.net/wp-content/uploads/2024/01/UKHRA2022_HRCS_public_dataset_v1-2_30Jan2024.xlsx
	wget -O data/raw/ukhra2018.xlsx https://hrcsonline.net/wp-content/uploads/2020/01/UKHRA2018_HRCS_public_dataset_v1_27Jan2020.xlsx
	wget -O data/raw/ukhra2014.xlsx https://hrcsonline.net/wp-content/uploads/2018/01/UK_Health_Research_Analysis_Data_2014_public_v1_27Oct2015.xlsx
	wget -O data/raw/nihr_all.parquet https://nihr.opendatasoft.com/api/explore/v2.1/catalog/datasets/nihr-summary-view/exports/parquet?lang=en&timezone=Europe%2FLondon
	@echo "Data download complete!"

.PHONY: build_dataset
build_dataset:
	uv run src/data_processing/data_processing.py

.PHONY: preprocess
preprocess:
	uv run src/preprocess.py \
		"config/train_config.yaml" \
		"data/clean/clean.parquet" \
		"data/preprocessed"

.PHONY: train
train:
	@if [ -z "${path}" ]; then \
        echo "Error: path variable is undefined - please specify as make tests path=<path>"; \
        exit 1; \
    fi
	@echo "Training data directory: ${path}"

	uv run src/train.py \
		--config-path "config/train_config.yaml" \
		--train-path "${path}/train_synthetic.parquet" \
		--test-path "${path}/test.parquet" \
		--label-names-path "data/label_names/ukhra_ra.jsonl" \
		--model-dir "data/model/"

.PHONY: train_ra
train_ra:
	$(MAKE) train path="data/preprocessed/ra"

.PHONY: train_ra_top
train_ra_top:
	$(MAKE) train path="data/preprocessed/ra_top"

.PHONY: train_hc
train_hc:
	$(MAKE) train path="data/preprocessed/hc"
