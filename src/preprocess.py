import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

TOP_RA_ONLY = True

def binarize(data: pd.DataFrame):
    """
    Binarize data.
    """
    mlb = MultiLabelBinarizer()
    if TOP_RA_ONLY:
        data['RA_bin'] = mlb.fit_transform(data['RA_top']).tolist()
    else:
        data['RA_bin'] = mlb.fit_transform(data['RA']).tolist()
    return data

def split_data_frame(data: pd.DataFrame, test_size=0.2):
    """
    Split data into training and test sets.
    """
    return train_test_split(data, test_size=test_size, random_state=42)

def clean_dataframe(data: pd.DataFrame):
    """
    Process data.
    """
    # rename columns 
    data.rename(columns={'AllText': 'text', 'RA_bin': 'label'}, inplace=True)
    # drop other columns
    data = data[['text', 'label']]
    return data

def save_train_test_data(train, test):
    """
    Save training and test data to disk.
    """
    if not os.path.exists('../data/preprocessed'):
        os.makedirs('../data/preprocessed')
    train.to_parquet('../data/preprocessed/train.parquet')
    test.to_parquet('../data/preprocessed/test.parquet')

if __name__ == "__main__":
    sys.path.append('../data')
    data = pd.read_parquet('../data/clean/ukhra_ra.parquet')

    data = binarize(data)
    train, test = split_data_frame(data)
    train = clean_dataframe(train)
    test = clean_dataframe(test)
    save_train_test_data(train, test)