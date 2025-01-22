# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# \
	Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

.PHONY: build_dataset
build_dataset:
	mkdir -p data
	mkdir -p data/raw
	mkdir -p data/clean
	wget -O data/raw/ukhra2022.xlsx https://hrcsonline.net/wp-content/uploads/2024/01/UKHRA2022_HRCS_public_dataset_v1-2_30Jan2024.xlsx
	wget -O data/raw/ukhra2018.xlsx https://hrcsonline.net/wp-content/uploads/2020/01/UKHRA2018_HRCS_public_dataset_v1_27Jan2020.xlsx
	wget -O data/raw/ukhra2014.xlsx https://hrcsonline.net/wp-content/uploads/2018/01/UK_Health_Research_Analysis_Data_2014_public_v1_27Oct2015.xlsx
	python src/data_processing.py

.PHONY: preprocess
preprocess:
	python src/preprocess.py --config "config/train_config.yaml" --clean-data "data/clean/ukhra_ra.parquet" --output-dir "data/preprocessed"

.PHONY: preprocess-cased
preprocess:
	python src/preprocess.py --config "config/train_config.yaml" --clean-data "data/clean/ukhra_ra.parquet" --output-dir "data/preprocessed" --cased

.PHONY: train
