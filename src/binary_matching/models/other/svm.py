import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


def process_datasets():
    train_df = pd.read_parquet('data/embeddings/train.parquet')
    train_x = np.array(train_df['embeddings'].to_list())
    train_y = train_df[list(train_df)[:-2]].to_numpy()

    test_df = pd.read_parquet('data/embeddings/test.parquet')
    test_x = np.array(test_df['embeddings'].to_list())
    test_y = test_df[list(test_df)[:-2]].to_numpy()

    svm = OneVsRestClassifier(SVC(), n_jobs=-1)
    svm.fit(train_x, train_y)

    y_hat = svm.predict(test_x)
    macro_f1 = f1_score(test_y, y_hat, average='macro')

    print(macro_f1)


if __name__ == '__main__':
    process_datasets()
