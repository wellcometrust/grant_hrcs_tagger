import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import yaml


def load_yaml_config(yaml_path: str):
    """ Load yaml configuration file.

    Returns:
        dict: configuration dictionary
    """
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def binarize(data: pd.DataFrame):
    """
    Binarize data.

    Args:
        data (pd.DataFrame): Dataframe to binarize.

    Returns:
        pd.DataFrame: dataframe with binarized labels.
    """
    mlb = MultiLabelBinarizer()
    category = config['training_settings']['category']
    if category == 'top_RA':
        data['RA_bin'] = mlb.fit_transform(data['RA_top']).tolist()
    elif category == 'RA':
        data['RA_bin'] = mlb.fit_transform(data['RA']).tolist()
    elif category == 'HC':
        data.dropna(subset=['HC'], inplace=True)
        data['HC_bin'] = mlb.fit_transform(data['HC']).tolist()
    return data


def split_data_frame(data: pd.DataFrame, test_size=0.2):
    """
    Split data into training and test sets.

    Args:
        data (pd.DataFrame): Dataframe to split.
        test_size (float): Size of the test set.

    Returns:
        tuple: training and test data
    """
    return train_test_split(data, test_size=test_size, random_state=42)


def clean_dataframe(data: pd.DataFrame, cased=False):
    """
    Changes labels and trims data to only include text and label columns.

    Args:
        data (pd.DataFrame): Dataframe to clean.

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # rename columns
    data.rename(
        columns={'AllText': 'text', 'RA_bin': 'label', 'HC_bin': 'label'},
        inplace=True
    )

    # drop other columns
    data = data[['text', 'label']].copy()
    # if cased is set to False, convert text to lowercase
    if not cased:
        data['text'] = data['text'].str.lower()

    return data


def save_train_test_data(train, test, value_counts, output_dir):
    """
    Save training and test data to disk.

    Args:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Test data.
        value_counts (dict): Mapping of label to count in the dataset.
        output_dir (str): Output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train.to_parquet(output_dir+'/train.parquet')
    test.to_parquet(output_dir+'/test.parquet')

    # save value_counts
    with open(output_dir+'/value_counts.json', 'w') as f:
        json.dump(value_counts, f)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='../config/train_config.yaml'
    )
    parser.add_argument(
        '--clean-data',
        type=str,
        default='../data/clean/ukhra_ra.parquet'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/preprocessed'
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    # pipeline
    data = pd.read_parquet(args.clean_data)
    data = binarize(data)
    train, test = split_data_frame(
        data,
        config['preprocess_settings']['test_train_split']
    )
    train = clean_dataframe(train, config['preprocess_settings']['cased'])
    test = clean_dataframe(test, config['preprocess_settings']['cased'])

    # calculate the value counts for each label (to compare evaluating metrics
    # against the distribution of labels)
    category = config['training_settings']['category']
    if category == 'top_RA':
        data_RA_exploded = data.explode('RA_top')
        value_counts = data_RA_exploded['RA_top'].value_counts()
    elif category == 'RA':
        data_RA_exploded = data.explode('RA')
        value_counts = data_RA_exploded['RA'].value_counts()
    elif category == 'HC':
        data_RA_exploded = data.explode('HC')
        value_counts = data_RA_exploded['HC'].value_counts()
    value_counts = value_counts.to_dict()

    # save data
    save_train_test_data(train, test, value_counts, args.output_dir)
