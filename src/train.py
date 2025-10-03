import argparse
from functools import partial
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    ModernBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import WandbCallback
import wandb
from utils import load_yaml_config

os.environ["WANDB_LOG_MODEL"] = "end"


def init_device():
    """Initialize device to use for training.

    Returns:
        str: device to use for training
    """
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")
    if torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


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


def train(train_data, test_data, model_path, config, class_counts, class_weighting, class_labels, test_counts):
    """Finetune a model from the config for the UKHRA data.

    Args:
        train_data (pd.DataFrame): Training dataframe
        test_data (pd.DataFrame): Test dataframe
        model_path (str): Path to save model.
        config (dict): Configuration dictionary.

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
    wandb.log({"device": device})

    num_labels = len(test_dataset[0]["labels"])
    wandb.log({"num_labels": num_labels})

    # initialize model
    if "modernbert" in config["training_settings"]["model"].lower():
        model = ModernBertForSequenceClassification.from_pretrained(
            config["training_settings"]["model"],
            num_labels=num_labels,
            problem_type="multi_label_classification",
            reference_compile=False,
        )

        print("model initialized using ModernBertForSequenceClassification")
    elif "distilbert" in config["training_settings"]["model"].lower():
        model = DistilBertForSequenceClassification.from_pretrained(
            config["training_settings"]["model"],
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        print("model initialized using DistilBertForSequenceClassification")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config["training_settings"]["model"],
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        print("model initialized using AutoModelForSequenceClassification")

    model.to(device)

    # initialize training arguments
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
        report_to=config["training_settings"]["report_to"],
        save_strategy=config["training_settings"]["save_strategy"],
        save_total_limit=config["training_settings"]["save_total_limit"],
        output_dir=model_path,
        logging_strategy=config["training_settings"]["logging_strategy"],
    )

    compute_metrics_fn = partial(compute_metrics, config=config)

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
            compute_metrics=compute_metrics_fn,
            class_weights=class_weights,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )

    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        val_dataset=test_dataset,
        num_samples=10,
        freq=1,
        config=config,
        class_labels=class_labels,
        train_count=class_counts,
        test_count=test_counts
    )

    # Add the callback to the trainer
    trainer.add_callback(progress_callback)

    # train and evaluate
    trainer.train()
    metrics = trainer.evaluate()

    # save model and tokenizer
    tokenizer.save_pretrained(model_path + "/tokenizer")
    trainer.save_model(output_dir=model_path)

    return metrics


def compute_metrics(eval_pred, config):
    """Compute evaluation metrics to be used in the Trainer.

    Args:
        eval_pred (tuple): Tuple containing logits and labels.
        config (dict): Configuration dictionary from yaml file.

    Returns:
        dict: Evaluation metrics (f1, f1_macro, f1_micro, precision, recall)
    """
    logits, labels = eval_pred
    # apply sigmoid to logits
    logits = torch.sigmoid(torch.tensor(logits)).cpu().detach().numpy()

    if config["training_settings"]["output_weighting"]:
        thresholds = [1] * labels.shape[1]
        category = config["training_settings"].get("category")
        if category in {"RA", "top_RA"}:
            # thresholds tuned for RA/top_RA category
            thresholds[:4] = [0.2, 0.5, 0.8, 0.95]
        else:
            thresholds[:4] = [0.2, 0.6, 0.8, 0.9]

        predictions = np.zeros_like(logits)
        for i, logit in enumerate(logits):
            sorted_indices = np.argsort(logit)[::-1]
            for rank, idx in enumerate(sorted_indices):
                if rank < len(thresholds) and logit[idx] > thresholds[rank]:
                    predictions[i, idx] = 1
    else:
        predictions = np.where(logits > 0.5, 1, 0)

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
    print(metrics)
    return metrics


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset
          for generating predictions.
        num_samples (int, optional): Number of samples to select from
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 1.
    """

    def __init__(self, trainer, tokenizer, val_dataset, config, class_labels, train_count, test_count, num_samples=100, freq=1):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 1.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq
        self.config = config
        self.class_labels = class_labels
        self.train_counts = train_count
        self.test_counts = test_count

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if state.epoch % self.freq == 0:
            predictions = self.trainer.predict(self.sample_dataset)
            metrics = compute_metrics(predictions, self.config)
            self._wandb.log({"f1_micro": metrics["f1_micro"],
                             "f1_macro": metrics["f1_macro"]})
            table = plot_metrics(
                metrics=metrics, 
                class_labels=self.class_labels, 
                train_counts=self.train_counts, 
                test_counts=self.test_counts
                )
            self._wandb.log({"metrics and value_count table": table})


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

    table = wandb.Table(
        data=[list(values) for values in data],
        columns=["label", "precision", "recall", "f1", "train_count", "test_count"],
    )
    return table


def run_training(args):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    model_name = config["training_settings"]["model"]
    model_path = f"{args.model_dir}/{model_name}_{timestamp}"
    # check if model path exists
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_data = pd.read_parquet(args.train_path)
    test_data = pd.read_parquet(args.test_path)

    class_labels = list(train_data.columns[:-1])
    with open(args.label_names_path) as f:
        label_names = {k: v for line in f for k, v in json.loads(line).items()}

    # add in the RA full name for reporting.
    if "RA" in config["training_settings"]["category"]:
        with open(args.label_names_path) as f:
            label_names = {k: v for line in f for k, v in json.loads(line).items()}
        class_labels = [f"{label}-{label_names[label]}" for label in class_labels]
    
    train_counts = np.sum(train_data[train_data.columns[:-1]].to_numpy(), axis=0)
    test_counts = np.sum(test_data[test_data.columns[:-1]].to_numpy(), axis=0)

    wandb.init(
        project=config["wandb_settings"]["project_name"],
        config=config["training_settings"],
    )

    wandb.log({"model_path": model_path})

    class_weighting = config["training_settings"]["class_weighting"]
    metrics = train(
        train_data,
        test_data,
        model_path=model_path,
        config=config,
        class_counts=train_counts,
        class_weighting=class_weighting,
        class_labels=class_labels,
        test_counts=test_counts,
    )

    table = plot_metrics(metrics, class_labels, train_counts, test_counts)
    wandb.log({"metrics and value_count table": table})


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config/train_config.yaml")
    dp = "data/preprocessed"
    parser.add_argument("--train-path", type=str, default=f"{dp}/train.parquet")
    parser.add_argument("--test-path", type=str, default=f"{dp}/test.parquet")
    parser.add_argument(
        "--label-names-path", type=str, default="data/label_names/ukhra_ra.jsonl"
    )
    parser.add_argument("--model-dir", type=str, default="data/model")
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    run_training(args)
