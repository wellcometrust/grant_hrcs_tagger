import numpy as np
from sklearn import metrics


def calculate_metrics(array):
    print('Binary matching metrics:')
    print('F1 score:', metrics.f1_score(array[:, 0], array[:, 1]))
    print('Accuracy:', metrics.accuracy_score(array[:, 0], array[:, 1]))

    print('Multi-label classification metrics:')
    for label in np.unique(array[:, 2]):
        label_data = array[array[:, 2] == label]

        print(
            f'{label} F1 score:',
            metrics.f1_score(label_data[:, 0], label_data[:, 1])
        )

        print(
            f'{label} Accuracy score:',
            metrics.accuracy_score(label_data[:, 0], label_data[:, 1]),
            '%'
        )
