import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb


def load_label_names(label_names_dir, label_type="long"):
    """Load label names from a JSON file.

    Args:
        label_names_dir (str): Path to the label names directory.
        label_type (str): Type of labels to load - "long" or "short".

    Returns:
        dict: Dictionary mapping label IDs to label names.
    """
    if label_type == "long":
        filepath = os.path.join(label_names_dir, "label_mapping_long.json")
    elif label_type == "short":
        filepath = os.path.join(label_names_dir, "label_mapping_short.json")
    elif label_type == "tiny":
        filepath = os.path.join(label_names_dir, "label_mapping_tiny.json")
    else:
        raise ValueError("label_type must be 'long', 'short', or 'tiny'")
        
    with open(filepath) as f:
        data = json.load(f)
        label_names = {idx: name for idx, name in enumerate(data.values())}
    
    return label_names

def get_parent_categories(predictions, labels, detailed_label_names, parent_level="parent"):
    """Extract parent category predictions and labels from detailed predictions.
    
    Args:
        predictions (np.array): Detailed level predictions
        labels (np.array): Detailed level labels  
        detailed_label_names (dict): Mapping of detailed label indices to names
        parent_level (str): "parent" for 8 categories, "super_parent" for 3 categories
        
    Returns:
        tuple: (parent_predictions, parent_labels) as np.arrays
    """
    # Create mapping from detailed labels to parent categories
    label_to_parent = {}
    for idx, label_name in detailed_label_names.items():
        try:
            parent_num = int(label_name.split(".")[0])
            label_to_parent[idx] = parent_num
        except (ValueError, AttributeError):
            # Handle any labels that don't follow the expected format
            label_to_parent[idx] = 0
    
    num_samples = predictions.shape[0]
    # Convert to parent category predictions (8 categories)
    if parent_level == "super_parent":
        num_parent_cats = 3
    else:
        num_parent_cats = 8
    parent_predictions = np.zeros((num_samples, num_parent_cats))
    parent_labels = np.zeros((num_samples, num_parent_cats))
    
    for sample_idx in range(num_samples):
        for label_idx in range(predictions.shape[1]):
            parent_cat = label_to_parent.get(label_idx, 0)
            if parent_level == "parent" and parent_cat > 0:
                idx = parent_cat - 1  # Convert to 0-based indexing
                
            elif parent_level == "super_parent":
                # Map parent categories to super parent categories
                if parent_cat in [1, 2, 3]:  # Super Parent 1
                    idx = 0
                elif parent_cat in [4, 5]:  # Super Parent 2
                    idx = 1
                elif parent_cat in [6, 7, 8]:  # Super Parent 3
                    idx = 2
                else:
                    continue  # Skip if no valid parent category
            else:
                continue  # Skip if no valid parent category

            if predictions[sample_idx, label_idx] == 1:
                parent_predictions[sample_idx, idx] = 1
            if labels[sample_idx, label_idx] == 1:
                parent_labels[sample_idx, idx] = 1
    
    return parent_predictions, parent_labels

def calculate_metrics(labels, predictions, prefix=""):
    """Calculate standard classification metrics.
    
    Args:
        labels (np.array): True labels
        predictions (np.array): Predicted labels
        prefix (str): Prefix to add to metric names (e.g., "parent_", "parent_wellcome_", "super_parent_*")
    Returns:
        dict: Dictionary of calculated metrics (arrays) plus per-class F1 entries.
    """
    # Choose label mapping based on prefix family
    if prefix.startswith("parent_"):
        label_names = load_label_names("data/label_names", "short")
    elif prefix.startswith("super_parent_"):
        label_names = load_label_names("data/label_names", "tiny")
    else:
        label_names = load_label_names("data/label_names", "long")
    label_names = list(label_names.values())

    f1_macro = f1_score(labels, predictions, average="macro")
    f1_micro = f1_score(labels, predictions, average="micro")
    f1_per_class = f1_score(labels, predictions, average=None)
    precision_macro = precision_score(labels, predictions, average="macro")
    precision_micro = precision_score(labels, predictions, average="micro")
    recall_macro = recall_score(labels, predictions, average="macro")
    recall_micro = recall_score(labels, predictions, average="micro")
    precision = precision_score(labels, predictions, average=None)
    recall = recall_score(labels, predictions, average=None)

    # Guard against length mismatches between mapping and predictions
    n_classes = predictions.shape[1]
    indexed_len = min(n_classes, len(label_names))

    # Per-class entries with readable label names (grouped for W&B)
    per_class_metrics = {
        # F1 per class
        f"{prefix}f1/{label_names[idx]}": float(f1_per_class[idx])
        for idx in range(indexed_len)
    } | {
        # Precision per class
        f"{prefix}precision/{label_names[idx]}": float(precision[idx])
        for idx in range(indexed_len)
    } | {
        # Recall per class
        f"{prefix}recall/{label_names[idx]}": float(recall[idx])
        for idx in range(indexed_len)
    }

    table = wandb.Table(
        columns=["prefix", "precision_macro", "precision_micro", "recall_macro", "recall_micro", "f1_macro", "f1_micro", "train_count", "test_count"],
        data=[[prefix,  
                float(precision_macro),
                float(precision_micro),
                float(recall_macro),
                float(recall_micro),
                float(f1_macro),
                float(f1_micro),
                int(np.sum(labels[:, 0])),
                int(np.sum(predictions[:, 0])),
            ]]
    )

    table_per_class = wandb.Table(
        columns=["label", "f1", "precision", "recall"],
        data=[
            [label_names[idx], 
             float(f1_per_class[idx]), 
             float(precision[idx]), 
             float(recall[idx])]
            for idx in range(indexed_len)
        ]
    )
    wandb.log({f"{prefix}per_class_metrics": table_per_class})
    wandb.log({f"{prefix}metrics and value_count table": table})

    return {
        f"{prefix}f1_macro": float(f1_macro),
        f"{prefix}f1_micro": float(f1_micro),
        f"{prefix}precision_macro": float(precision_macro),
        f"{prefix}precision_micro": float(precision_micro),
        f"{prefix}recall_macro": float(recall_macro),
        f"{prefix}recall_micro": float(recall_micro),
        **per_class_metrics,
    }


def prepare_compute_metrics(config, meta_path: str | None = None):
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
            recall) for both detailed and parent categories
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

        # Calculate detailed level metrics
        detailed_metrics = calculate_metrics(labels, predictions)
        
        # Parent category metrics
        try:
            detailed_label_names = load_label_names("data/label_names", "long")
            parent_predictions, parent_labels = get_parent_categories(
                predictions, labels, detailed_label_names
            )
            parent_metrics = calculate_metrics(
                parent_labels, parent_predictions, prefix="parent_"
            )
            # Super parent metrics (3 categories) for full eval set
            super_parent_predictions, super_parent_labels = get_parent_categories(
                predictions, labels, detailed_label_names, parent_level="super_parent"
            )
            super_parent_metrics = calculate_metrics(
                super_parent_labels, super_parent_predictions, prefix="super_parent_"
            )
            all_metrics = {**detailed_metrics, **parent_metrics, **super_parent_metrics}
        except Exception as e:
            print(f"Warning: Could not calculate parent metrics: {e}")
            all_metrics = detailed_metrics

        # Optional: Wellcome Trust filtered metrics using meta_path directory
        try:
            if not meta_path:
                print("Info: No meta_path provided; skipping Wellcome metrics.")
                wandb.log({"wellcome_metrics_status": "no_meta_path_provided"})
            else:
                test_meta_path = os.path.join(meta_path, "test_meta.parquet")
                if not os.path.exists(test_meta_path):
                    print("Info: test_meta.parquet not found. Run train_test_split.py to generate metadata.")
                    wandb.log({"wellcome_metrics_status": "meta_not_found"})
                else:
                    meta_df = pd.read_parquet(test_meta_path)
                    if "FundingOrganisation" not in meta_df.columns:
                        print("Info: FundingOrganisation column missing in meta; skipping Wellcome metrics.")
                        wandb.log({"wellcome_metrics_status": "funding_org_missing"})
                    elif len(meta_df) != labels.shape[0]:
                        print("Info: test_meta length does not match eval set; skipping Wellcome metrics.")
                        wandb.log({"wellcome_metrics_status": "length_mismatch"})
                    else:
                        wt_mask = meta_df["FundingOrganisation"].astype(str).eq("Wellcome Trust").to_numpy()
                        if wt_mask.any():
                            labels_wt = labels[wt_mask]
                            predictions_wt = predictions[wt_mask]
                            # low level detailed metrics for Wellcome Trust filtered rows
                            wt_detailed = calculate_metrics(labels_wt, predictions_wt, prefix="wellcome_")

                            # parent level metrics for Wellcome Trust filtered rows
                            parent_pred_wt, parent_labels_wt = get_parent_categories(
                                predictions_wt, labels_wt, detailed_label_names
                            )
                            wt_parent = calculate_metrics(
                                parent_labels_wt, parent_pred_wt, prefix="parent_wellcome_"
                            )

                            # super parent level metrics for Wellcome Trust filtered rows
                            super_parent_pred_wt, super_parent_labels_wt = get_parent_categories(
                                predictions_wt, labels_wt, detailed_label_names, parent_level="super_parent"
                            )
                            wt_super_parent = calculate_metrics(
                                super_parent_labels_wt, super_parent_pred_wt, prefix="super_parent_wellcome_"
                            )
                            all_metrics.update({**wt_detailed, **wt_parent, **wt_super_parent})
                            wandb.log({"wellcome_metrics_status": "computed"})
                        else:
                            print("Info: No rows for FundingOrganisation == 'Wellcome Trust' in eval set")
                            wandb.log({"wellcome_metrics_status": "no_wellcome_rows"})
        except Exception as e:
            print(f"Warning: Could not compute Wellcome Trust filtered metrics: {e}")
        
        return all_metrics

    return compute_metrics

