import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb


def load_label_names(label_names_path):
    """Load label names from a JSONL file.

    Args:
        label_names_path (str): Path to the label names file.

    Returns:
        dict: Dictionary mapping label IDs to label names.
    """

    with open(label_names_path) as f:
        label_names = {k: v for line in f for k, v in json.loads(line).items()}
        label_names = {idx: name for idx, name in enumerate(label_names.values())}

    return label_names

def prepare_compute_metrics(config):
    """Wrapper for compute_metrics so config can be accessed

    Args: config (dict): Configuration dictionary from yaml file.

    Returns:
        function: compute_metrics function
    """

    def compute_metrics(eval_pred):
        """Compute evaluation metrics to be used in the Trainer.

        Args: eval_pred (tuple): Tuple containing logits and labels.

        Returns:
            dict: Evaluation metrics (f1, f1_macro, f1_micro, precision,
            recall)
        """
        logits, labels = eval_pred
        # apply sigmoid to logits
        logits = torch.sigmoid(torch.tensor(logits)).cpu().detach().numpy()

        num_tags_predicted = []
        if config["training_settings"]["output_weighting"]:
            thresholds = [1] * labels.shape[1]
            output_thresholds = config["training_settings"].get("output_thresholds", [0.2, 0.5, 0.8, 0.95])
            thresholds[0:len(output_thresholds)] = output_thresholds

            # Prepare an array to hold your predictions
            predictions = np.zeros_like(logits)

            # Loop through each sample's logits
            for i, logit in enumerate(logits):
                # Get the indices of the logits sorted by value in descending order
                sorted_indices = np.argsort(logit)[::-1]

                # Assign 1 to the top logits that exceed their respective thresholds
                for rank, idx in enumerate(sorted_indices):
                    if logit[idx] > thresholds[rank]:
                        predictions[i, idx] = 1
                num_tags_predicted.append(np.sum(predictions[i]))
        else:
            predictions = np.where(logits > 0.5, 1, 0)
            for prediction in predictions:
                num_tags_predicted.append(np.sum(prediction))

            print(
                "num_tags_predicted: ",
                pd.Series(num_tags_predicted).value_counts().sort_index(),
            )
        
        log_data = pd.Series(num_tags_predicted).value_counts()
        wandb.log({"num_tags_predicted": log_data.sort_index().to_json()})

        # compute actual metrics
        f1_macro = f1_score(labels, predictions, average="macro")
        f1_micro = f1_score(labels, predictions, average="micro")
        f1 = f1_score(labels, predictions, average=None)
        precision = precision_score(labels, predictions, average=None)
        recall = recall_score(labels, predictions, average=None)
        metrics = {
            "f1": f1,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "precision": precision,
            "recall": recall,
        }
        
        wandb.log({"metrics": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}})
        return metrics

    return compute_metrics

def plot_metrics(metrics, class_labels, train_counts, test_counts):
    """Plot evaluation metrics and value counts in wandb

    Args:
        metrics (dict): Evaluation metrics.
        class_labels (list): List of class labels.
        train_counts (list): List of class counts for train dataset.
        test_counts (list): List of class counts for test dataset.

    """

    f1 = metrics["eval_f1"]
    precision = metrics["eval_precision"]
    recall = metrics["eval_recall"]
    data = zip(
        class_labels, precision, recall, f1, train_counts, test_counts, strict=False
    )

    # print data as a dataframe to console
    df = pd.DataFrame(
        data,
        columns=["label", "precision", "recall", "f1", "train_count", "test_count"],
    )
    print(df)

    table = wandb.Table(
        data=[list(values) for values in data],
        columns=["label", "precision", "recall", "f1", "train_count", "test_count"],
    )

    wandb.log({"metrics and value_count table": table})
