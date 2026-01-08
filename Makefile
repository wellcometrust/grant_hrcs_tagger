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
	python src/data_processing/data_processing.py

.PHONY: train_test_split	
train_test_split:
	python src/train_test_split.py \
		"config/train_config.yaml" \
		"data/clean/clean.parquet" \
		"data/preprocessed"

.PHONY: train
train:
	@if [ -z "${data_path}" ]; then \
        echo "Error: path variable is undefined - please specify as make tests path=<path>"; \
        exit 1; \
    fi
	@echo "Training data directory: ${data_path}"

	python src/train.py \
		--config-path "config/train_config.yaml" \
		--train-path "${data_path}/train.parquet" \
		--test-path "${data_path}/test.parquet" \
		--label-names-dir "data/label_names/" \
		--model-dir "data/model/"

.PHONY: train_ra
train_ra:
	$(MAKE) train data_path="data/preprocessed/ra"

.PHONY: train_ra_top
train_ra_top:
	$(MAKE) train data_path="data/preprocessed/ra_top"
