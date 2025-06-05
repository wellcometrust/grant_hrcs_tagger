import os

import click
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split


def load_yaml_config(yaml_path: str):
    """Load yaml configuration file.

    Returns:
        dict: configuration dictionary
    """
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def split_data_frame(df: pd.DataFrame, category: str, test_size=0.2):
    """Split data into training and test sets.

    Args:
        df (pd.DataFrame): Dataframe to split.
        test_size (float): Size of the test set.

    Returns:
        tuple: training and test data

    """
    df.dropna(subset=category, inplace=True)

    # Randomly shuffle dataframe.
    df = df.sample(frac=1, random_state=10)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df[category])
    X = np.array([df["AllText"].to_numpy()]).T

    np.random.seed(5)
    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X, y, test_size=test_size
    )

    train = pd.DataFrame(y_train, columns=mlb.classes_)
    test = pd.DataFrame(y_test, columns=mlb.classes_)

    train["text"] = X_train.ravel()
    test["text"] = X_test.ravel()

    return train, test


def save_train_test_data(train, test, output_dir):
    """Save training and test data to disk.

    Args:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Test data.
        output_dir (str): Output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train.to_parquet(output_dir + "/train.parquet")
    test.to_parquet(output_dir + "/test.parquet")


@click.command()
@click.argument("config")
@click.argument("clean_data")
@click.argument("output_dir")
def processing_pipeline(config, clean_data, output_dir):
    """Process data into train, test datasets.

    Args:
        config(str): Path to config file.
        clean_data(str): Location of clean data parquet.
        output_dir(str): Base output directory path.

    """
    config = load_yaml_config(config)
    for category in ["RA", "RA_top", "HC"]:
        data = pd.read_parquet(clean_data)

        if not config["preprocess_settings"]["cased"]:
            data["AllText"] = data["AllText"].str.lower()

        train, test = split_data_frame(
            data, category, config["preprocess_settings"]["test_train_split"]
        )

        output_sub_dir = f"{output_dir}/{category.lower()}"
        save_train_test_data(train, test, output_sub_dir)


if __name__ == "__main__":
    processing_pipeline()
