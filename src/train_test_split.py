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
    """Split data into training and test sets and return aligned metadata.

    Args:
        df (pd.DataFrame): Dataframe to split.
        test_size (float): Size of the test set.

    Returns:
        tuple: (train_df, test_df, train_meta_df, test_meta_df)

    Notes:
        - train_df/test_df keep original schema: label columns + final 'text' column.
        - metadata is saved separately to avoid breaking training code that
          assumes 'text' is the last column and all prior columns are labels.
    """
    df = df.copy()
    df.dropna(subset=category, inplace=True)

    # Randomly shuffle dataframe and reset index; add a stable row id
    df = df.sample(frac=1, random_state=10).reset_index(drop=True)
    df["_row_id"] = np.arange(len(df), dtype=np.int64)

    # Labels (multilabel binarized)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df[category])

    # Build X with two columns: text and row_id to keep perfect alignment
    X = np.column_stack((df["AllText"].to_numpy(), df["_row_id"].to_numpy()))

    np.random.seed(5)
    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X, y, test_size=test_size
    )

    # Extract text and indices from the split X
    text_train = X_train[:, 0]
    text_test = X_test[:, 0]
    idx_train = X_train[:, 1].astype(np.int64)
    idx_test = X_test[:, 1].astype(np.int64)

    # Compose label+text DataFrames (text must remain the last column)
    train = pd.DataFrame(y_train, columns=mlb.classes_)
    test = pd.DataFrame(y_test, columns=mlb.classes_)
    train["text"] = text_train
    test["text"] = text_test

    # Prepare sidecar metadata DataFrames aligned row-by-row to train/test
    metadata_cols = [c for c in [
        "FundingOrganisation",
        "OrganisationReference"
    ] if c in df.columns]

    if metadata_cols:
        df_meta_indexed = df.set_index("_row_id")
        train_meta = df_meta_indexed.loc[idx_train, metadata_cols].reset_index(drop=True)
        test_meta = df_meta_indexed.loc[idx_test, metadata_cols].reset_index(drop=True)
    else:
        train_meta = pd.DataFrame(index=np.arange(len(train)))
        test_meta = pd.DataFrame(index=np.arange(len(test)))

    return train, test, train_meta, test_meta


def save_train_test_data(train, test, output_dir, train_meta=None, test_meta=None):
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
    if train_meta is not None:
        train_meta.to_parquet(output_dir + "/train_meta.parquet")
    if test_meta is not None:
        test_meta.to_parquet(output_dir + "/test_meta.parquet")


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

        train, test, train_meta, test_meta = split_data_frame(
            data, category, config["preprocess_settings"]["test_train_split"]
        )

        output_sub_dir = f"{output_dir}/{category.lower()}"
        save_train_test_data(train, test, output_sub_dir, train_meta, test_meta)


if __name__ == "__main__":
    processing_pipeline()
