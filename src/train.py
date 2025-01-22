import json
import os
import sys
import wandb
import yaml

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (Trainer, TrainingArguments, AutoTokenizer,
                          DataCollatorWithPadding, AutoModelForSequenceClassification)


def load_yaml_config():
    with open('../config/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def init_device():
    if torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

class HRCSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].astype(np.float32).tolist()
        return item

    def __len__(self):
        return len(self.labels)

def train(train_data_path, test_data_path, model_path, config):
    train_data = pd.read_parquet(train_data_path)
    test_data = pd.read_parquet(test_data_path)

    tokenizer = AutoTokenizer.from_pretrained(config['training_settings']['model'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_encoding = tokenizer(train_data['text'].tolist(), truncation=True, padding=True)
    test_encoding = tokenizer(test_data['text'].tolist(), truncation=True, padding=True)

    train_dataset = HRCSDataset(train_encoding, train_data['label'].tolist())
    test_dataset = HRCSDataset(test_encoding, test_data['label'].tolist())

    device = init_device()
    print(f"using device ", device)
    wandb.log({"device": device})
    
    num_labels = len(test_dataset[0]['labels'])
    print(f"found {num_labels} labels")
    wandb.log({"num_labels": num_labels})

    model = AutoModelForSequenceClassification.from_pretrained(config['training_settings']['model'], num_labels=num_labels, problem_type="multi_label_classification")
    model.to(device)

    training_args = TrainingArguments(
        learning_rate=config['training_settings']['learning_rate'],
        num_train_epochs=config['training_settings']['num_train_epochs'],
        per_device_train_batch_size=config['training_settings']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training_settings']['per_device_eval_batch_size'],
        weight_decay=config['training_settings']['weight_decay'],
        report_to=config['training_settings']["report_to"],
        output_dir=model_path,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics=trainer.evaluate()

    tokenizer.save_pretrained(model_path+"/tokenizer")
    trainer.save_model(output_dir=model_path)

    return metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # apply sigmoid to logits
    logits = torch.sigmoid(torch.tensor(logits)).cpu().detach().numpy()
    predictions = np.where(logits > 0.5, 1, 0)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_micro = f1_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average=None)
    precision = precision_score(labels, predictions, average=None)
    recall = recall_score(labels, predictions, average=None)
    return {"f1": f1, "f1_macro": f1_macro, "f1_micro": f1_micro, "precision": precision, "recall": recall}

def plot_metrics(metrics, RA_value_counts):

    # plot f1 per label
    f1 = metrics['eval_f1']
    precision = metrics['eval_precision']
    recall = metrics['eval_recall']
    data = [[label, precision, recall, f1, value_count] for label, precision, recall, f1, value_count in zip(RA_value_counts.keys(), f1, precision, recall, RA_value_counts.values())]
    table = wandb.Table(data=data, columns=["label", "precision", "recall", "f1", "value_count"])
    wandb.log({"metrics and value_count table":table})


if __name__ == "__main__":
    sys.path.append('../data')
    train_path = '../data/preprocessed/train.parquet'
    test_path = '../data/preprocessed/test.parquet'
    RA_value_counts_path = '../data/preprocessed/RA_value_counts.json'
    label_names_path = '../data/label_names/ukhra_ra.jsonl'

    config = load_yaml_config()
    wandb.init(project=config['wandb_settings']['project_name'])
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    model_name = config['training_settings']['model']
    model_path = f'../data/model/{model_name}_{timestamp}'
    # check if model path exists
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # pull in label names and value counts
    with open(RA_value_counts_path, 'r') as f:
        RA_value_counts = json.load(f)

    with open(label_names_path, 'r') as f:
        label_names = {k: v for line in f for k, v in json.loads(line).items()}

    RA_value_counts = {label+"-"+label_names[label]: count for label, count in RA_value_counts.items()}
    print(RA_value_counts)

    metrics = train(train_data_path=train_path, test_data_path=test_path, model_path=model_path, config=config)
    plot_metrics(metrics, RA_value_counts)