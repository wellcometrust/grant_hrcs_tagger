import numpy as np
import pandas as pd
from rich.progress import track
from scipy.spatial import distance

criteria_df = pd.read_parquet('data/embeddings/criteria.parquet')


class CosineDistances:
    def __init__(self):
        criteria_df = pd.read_parquet('data/embeddings/criteria.parquet')
        self.criteria_embeddings = criteria_df['embeddings'].tolist()

    def get_distances(self, grant):
        distances = []
        for criteria in self.criteria_embeddings:
            distances.append(distance.cosine(grant, criteria))

        return np.array(distances)


def get_grant_distances(grant):
    distances = []
    cosine_distances = CosineDistances()
    for grant in track(grant['embeddings']):
        distances.append(cosine_distances.get_distances(grant))

    return distances


def process_datasets():
    train_df = pd.read_parquet('data/embeddings/train.parquet')
    test_df = pd.read_parquet('data/embeddings/test.parquet')

    train_df['distances'] = get_grant_distances(train_df)
    test_df['distances'] = get_grant_distances(test_df)

    train_df.to_parquet('data/embeddings/train_cosine.parquet')
    test_df.to_parquet('data/embeddings/test_cosine.parquet')


if __name__ == '__main__':
    process_datasets()
