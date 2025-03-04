import argparse
import json
import os
import wandb
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    DistilBertForSequenceClassification,
    ModernBertForSequenceClassification
)


def load_yaml_config(config_path: str):
    """ Load yaml configuration file.

    Returns:
        dict: configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def init_device():
    """ Initialize device to use for training.

    Returns:
        str: device to use for training
    """
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    if torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class WeightedTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            class_weights = class_weights.to(self.args.device)

        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        """ How the loss is computed by Trainer. By default, all models return
        the loss in the first element. Subclass and override for custom
        behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        try:
            loss = self.loss_fct(
                outputs.logits.view(-1, model.num_labels),
                labels.view(-1, model.num_labels)
            )
        except AttributeError:  # DataParallel
            loss = self.loss_fct(
                outputs.logits.view(-1, model.module.num_labels),
                labels.view(-1, model.num_labels)
            )

        return (loss, outputs) if return_outputs else loss


class HRCSDataset(torch.utils.data.Dataset):
    """
    A custom Dataset class for handling encodings and labels for training and
    evaluation.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }

        item['labels'] = self.labels[idx].astype(np.float32).tolist()
        return item

    def __len__(self):
        return len(self.labels)


def train(
        train_data_path,
        test_data_path,
        model_path,
        config,
        value_counts,
        class_weighting):
    """
    Finetune a model from the config for the UKHRA data.

    Args:
        train_data_path (str): Path to training data.
        test_data_path (str): Path to test data.
        model_path (str): Path to save model.
        config (dict): Configuration dictionary.

    Returns:
        dict: Evaluation metrics.
    """
    # tokenize data and create datasets
    train_data = pd.read_parquet(train_data_path)
    test_data = pd.read_parquet(test_data_path)

    tokenizer = AutoTokenizer.from_pretrained(
        config['training_settings']['model']
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_encoding = tokenizer(
        train_data['text'].tolist(),
        truncation=True,
        padding=True
    )

    test_encoding = tokenizer(
        test_data['text'].tolist(),
        truncation=True,
        padding=True
    )

    train_y = [r for r in train_data[train_data.columns[:-1]].to_numpy()]
    test_y = [r for r in test_data[test_data.columns[:-1]].to_numpy()]

    train_dataset = HRCSDataset(train_encoding, train_y)
    test_dataset = HRCSDataset(test_encoding, test_y)

    device = init_device()
    wandb.log({"device": device})

    num_labels = len(test_dataset[0]['labels'])
    wandb.log({"num_labels": num_labels})

    # initialize model
    if 'modernbert' in config['training_settings']['model'].lower():
        model = ModernBertForSequenceClassification.from_pretrained(
            config['training_settings']['model'],
            num_labels=num_labels,
            problem_type="multi_label_classification",
            reference_compile=False
        )

        print("model initialized using ModernBertForSequenceClassification")
    elif 'distilbert' in config['training_settings']['model'].lower():
        model = DistilBertForSequenceClassification.from_pretrained(
            config['training_settings']['model'],
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        print("model initialized using DistilBertForSequenceClassification")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            config['training_settings']['model'],
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        print("model initialized using AutoModelForSequenceClassification")

    model.to(device)

    # initialize training arguments
    training_args = TrainingArguments(
        learning_rate=config['training_settings']['learning_rate'],
        num_train_epochs=config['training_settings']['num_train_epochs'],
        per_device_train_batch_size=config['training_settings'][
            'per_device_train_batch_size'
        ],
        per_device_eval_batch_size=config['training_settings'][
            'per_device_eval_batch_size'
        ],
        weight_decay=config['training_settings']['weight_decay'],
        report_to=config['training_settings']["report_to"],
        save_strategy=config['training_settings']['save_strategy'],
        save_total_limit=config['training_settings']['save_total_limit'],
        output_dir=model_path,
        logging_strategy=config['training_settings']['logging_strategy'],
    )

    # sort value_counts by key
    value_counts = {
        k: v for k, v in sorted(value_counts.items(), key=lambda item: item[0])
    }

    HRCS_values = list(value_counts.values())
    HRCS_values = [value/sum(HRCS_values) for value in HRCS_values]
    HRCS_values = [1/value for value in HRCS_values]

    # set class weights
    class_weights = torch.tensor(HRCS_values, dtype=torch.float32).to(device)

    compute_metrics = prepare_compute_metrics(config)
    # initialize trainer depending on class weighting option
    if class_weighting:
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights
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
    tokenizer.save_pretrained(model_path+"/tokenizer")
    trainer.save_model(output_dir=model_path)

    return metrics


def prepare_compute_metrics(config):
    """ Wrapper for compute_metrics so config can be accessed

    Args: config (dict): Configuration dictionary from yaml file.

    Returns:
        function: compute_metrics function
    """
    def compute_metrics(eval_pred):
        """ Compute evaluation metrics to be used in the Trainer.

        Args: eval_pred (tuple): Tuple containing logits and labels.

        Returns:
            dict: Evaluation metrics (f1, f1_macro, f1_micro, precision,
            recall)
        """
        logits, labels = eval_pred
        # apply sigmoid to logits
        logits = torch.sigmoid(torch.tensor(logits)).cpu().detach().numpy()

        num_tags_predicted = []
        if config['training_settings']['output_weighting']:
            thresholds = [1] * labels.shape[1]
            if (
                config['training_settings']['category'] == 'RA' or
                config['training_settings']['category'] == 'top_RA'
            ):
                # make a list of increasing thresholds same length as the
                # number of labels
                thresholds[0] = 0.2
                thresholds[1] = 0.5
                thresholds[2] = 0.8
                thresholds[3] = 0.95
            else:
                thresholds[0] = 0.2
                thresholds[1] = 0.6
                thresholds[2] = 0.8
                thresholds[3] = 0.9

            # Prepare an array to hold your predictions
            predictions = np.zeros_like(logits)

            # Loop through each sample's logits
            for i, logit in enumerate(logits):
                # Get the indices of the logits sorted by value in descending
                # order
                sorted_indices = np.argsort(logit)[::-1]

                # Assign 1 to the top logits that exceed their respective
                # thresholds
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
                pd.Series(num_tags_predicted).value_counts().sort_index()
            )
            log_data = pd.Series(num_tags_predicted).value_counts()
            wandb.log(
                {"num_tags_predicted": log_data.sort_index().to_json()}
            )

        # compute actual metrics
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_micro = f1_score(labels, predictions, average='micro')
        f1 = f1_score(labels, predictions, average=None)
        precision = precision_score(labels, predictions, average=None)
        recall = recall_score(labels, predictions, average=None)
        print(
            {
                "f1": f1,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "precision": precision,
                "recall": recall}
            )
        return {
            "f1": f1,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "precision": precision,
            "recall": recall
        }

    return compute_metrics


def plot_metrics(metrics, value_counts):
    """ Plot evaluation metrics and value counts in wandb

    Args:
        metrics (dict): Evaluation metrics.
        value_counts (dict): Mapping of label to count in the dataset
    """
    # plot f1 per label
    f1 = metrics['eval_f1']
    precision = metrics['eval_precision']
    recall = metrics['eval_recall']
    # order metrics to align with value_counts index
    value_count_keys = list(value_counts.keys())
    metric_idx = [value_count_keys.index(label) for label in value_count_keys]
    f1 = [f1[idx] for idx in metric_idx]
    precision = [precision[idx] for idx in metric_idx]
    recall = [recall[idx] for idx in metric_idx]

    data = [
        [
            label, precision, recall, f1, value_count
        ] for label, precision, recall, f1, value_count in zip(
            value_counts.keys(),
            precision,
            recall,
            f1,
            value_counts.values()
        )
    ]

    table = wandb.Table(
        data=data,
        columns=["label", "precision", "recall", "f1", "value_count"]
    )

    wandb.log({"metrics and value_count table": table})


def run_training(args):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    model_name = config['training_settings']['model']
    model_path = f'{args.model_dir}/{model_name}_{timestamp}'
    # check if model path exists
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # pull in label names and value counts
    with open(args.value_counts_path, 'r') as f:
        value_counts = json.load(f)

    with open(args.label_names_path, 'r') as f:
        label_names = {k: v for line in f for k, v in json.loads(line).items()}

    # add in the RA full name for reporting.
    if 'RA' in config['training_settings']['category']:
        value_counts = {
            label+"-"+label_names[label]: count for label,
            count in value_counts.items()
        }

    wandb.init(
        project=config['wandb_settings']['project_name'],
        config=config['training_settings']
    )

    wandb.log({"model_path": model_path})

    class_weighting = config['training_settings']['class_weighting']
    metrics = train(
        train_data_path=args.train_path,
        test_data_path=args.test_path,
        model_path=model_path,
        config=config,
        value_counts=value_counts,
        class_weighting=class_weighting
    )

    plot_metrics(metrics, value_counts)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='config/train_config.yaml')
    dp = 'data/preprocessed'
    parser.add_argument('--train-path', type=str, default=f'{dp}/train.parquet')
    parser.add_argument('--test-path', type=str, default=f'{dp}/test.parquet')
    parser.add_argument(
        '--value-counts-path',
        type=str,
        default=f'{dp}/value_counts.json'
    )
    parser.add_argument(
        '--label-names-path',
        type=str,
        default='data/label_names/ukhra_ra.jsonl'
    )
    parser.add_argument('--model-dir', type=str, default='data/model')
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    run_training(args)
