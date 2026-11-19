import argparse
import os

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from utils import load_yaml_config
from metrics import prepare_compute_metrics, load_label_names


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.FloatTensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)

        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        """How the loss is computed by Trainer. By default, all models return
        the loss in the first element. Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        try:
            loss = self.loss_fct(
                outputs.logits.view(-1, model.num_labels),
                labels.view(-1, model.num_labels),
            )
        except AttributeError:  # DataParallel
            loss = self.loss_fct(
                outputs.logits.view(-1, model.module.num_labels),
                labels.view(-1, model.num_labels),
            )

        return (loss, outputs) if return_outputs else loss


class HRCSDataset(torch.utils.data.Dataset):
    """A custom Dataset class for handling encodings and labels for training and
    evaluation.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        item["labels"] = self.labels[idx].astype(np.float32).tolist()
        return item

    def __len__(self):
        return len(self.labels)


def train(train_data, test_data, model_path, config, class_counts, class_weighting, label_names_dir, meta_path):
    """Finetune a model from the config for the UKHRA data.

    Args:
        train_data (pd.DataFrame): Training dataframe
        test_data (pd.DataFrame): Test dataframe
        model_path (str): Path to save model.
        config (dict): Configuration dictionary.
        class_counts (list): List of class counts for training data.
        class_weighting (bool): Whether to use class weighting.
        label_names_dir (str): Directory containing label names files.
        meta_path (str): Directory containing train/test metadata files.

    Returns:
        dict: Evaluation metrics.
    """
    # tokenize data and create datasets
    tokenizer = AutoTokenizer.from_pretrained(config["training_settings"]["model"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_encoding = tokenizer(
        train_data["text"].tolist(), truncation=True, padding=True
    )

    test_encoding = tokenizer(test_data["text"].tolist(), truncation=True, padding=True)

    train_y = [r for r in train_data[train_data.columns[:-1]].to_numpy()]
    test_y = [r for r in test_data[test_data.columns[:-1]].to_numpy()]

    train_dataset = HRCSDataset(train_encoding, train_y)
    test_dataset = HRCSDataset(test_encoding, test_y)

    device = init_device()
    log_to_wandb({"device": device}, config["training_settings"]["report_to"])

    num_labels = len(test_dataset[0]["labels"])
    log_to_wandb({"num_labels": num_labels}, config["training_settings"]["report_to"])

    # fetch label names
    label_names = load_label_names(label_names_dir, "long")

    # initialize model
    model = get_model(
        config["training_settings"]["model"],
        num_labels,
        label_names
    )
    if "modernbert" in config["training_settings"]["model"].lower():
        model.config.reference_compile=False

    # initialize training arguments
    report_to_value = config["training_settings"]["report_to"]
    if report_to_value is None:
        report_to_value = "none"  # Transformers expects "none" instead of None

    training_args = TrainingArguments(
        learning_rate=config["training_settings"]["learning_rate"],
        num_train_epochs=config["training_settings"]["num_train_epochs"],
        per_device_train_batch_size=config["training_settings"][
            "per_device_train_batch_size"
        ],
        per_device_eval_batch_size=config["training_settings"][
            "per_device_eval_batch_size"
        ],
        weight_decay=config["training_settings"]["weight_decay"],
        report_to=report_to_value,
        save_strategy=config["training_settings"]["save_strategy"],
        save_total_limit=config["training_settings"]["save_total_limit"],
        output_dir=model_path,
        logging_strategy=config["training_settings"]["logging_strategy"],
    )

    # Provide meta_path (directory holding train_meta/test_meta) to metrics
    compute_metrics = prepare_compute_metrics(config, meta_path)
    # initialize trainer depending on class weighting option
    if class_weighting:
        total_count = sum(class_counts)
        HRCS_values = [1 / (value / total_count) for value in class_counts]
        class_weights = torch.tensor(HRCS_values, dtype=torch.float32).to(device)

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    # train and evaluate
    trainer.train()
    metrics = trainer.evaluate()

    # save model and tokenizer
    tokenizer.save_pretrained(model_path + "/tokenizer")
    trainer.save_model(output_dir=model_path)

    return metrics


@click.command()
@click.argument('config-path')
@click.argument('train-path')
@click.argument('test-path')
@click.argument('label-names-dir')
@click.argument('model-dir')
def run_training(config_path, train_path, test_path, label_names_dir, model_dir):
    config = load_yaml_config(config_path)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

    model_name = config["training_settings"]["model"]
    model_path = f"{model_dir}/{model_name}_{timestamp}"
    os.makedirs(model_path)

    train_data = pd.read_parquet(train_path)
    test_data = pd.read_parquet(test_path)

    class_labels_dict = load_label_names(label_names_dir, "long")
    class_labels = [label for label in class_labels_dict.values()]
    print(f"class labels: {class_labels}")
    train_counts = np.sum(train_data[train_data.columns[:-1]].to_numpy(), axis=0)
    test_counts = np.sum(test_data[test_data.columns[:-1]].to_numpy(), axis=0)

    init_wandb(
        config["wandb_settings"]["project_name"],
        config["training_settings"],
        config["training_settings"]["report_to"]
    )

    log_to_wandb({"model_path": model_path}, config["training_settings"]["report_to"])

    class_weighting = config["training_settings"]["class_weighting"]
    meta_path = os.path.dirname(test_path)
    metrics = train(
        train_data,
        test_data,
        model_path=model_path,
        config=config,
        class_counts=train_counts,
        class_weighting=class_weighting,
        label_names_dir=label_names_dir,
        meta_path=meta_path,
    )


if __name__ == "__main__":
    run_training()
