from sklearn.metrics import f1_score
import os
import torch
import sys
import wandb
from torch import cuda
import numpy as np
import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, AutoModelForSequenceClassification, DistilBertTokenizerFast

MODEL = "distilbert-base-uncased"
wandb.init(project='grant_hrcs_tagger')

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

def train(train_data_path, test_data_path, model_path):
    train_data = pd.read_parquet(train_data_path)
    test_data = pd.read_parquet(test_data_path)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_encoding = tokenizer(train_data['text'].tolist(), truncation=True, padding=True)
    test_encoding = tokenizer(test_data['text'].tolist(), truncation=True, padding=True)

    train_dataset = HRCSDataset(train_encoding, train_data['label'].tolist())
    test_dataset = HRCSDataset(test_encoding, test_data['label'].tolist())

    if torch.backends.mps.is_built():
        device = "mps"
    elif cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"using device ", device)
    
    num_labels = len(test_dataset[0]['labels'])
    print(f"found {num_labels} labels")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels, problem_type="multi_label_classification")
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        learning_rate=2e-5,
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        weight_decay=0.01,               # strength of weight decay
        # logging_dir='./logs',            # directory for storing logs
        # logging_steps=10,
        report_to="wandb"
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
    wandb.log(metrics)

    tokenizer.save_pretrained(model_path+"/tokenizer")
    trainer.save_model(output_dir=model_path)


def compute_metrics(eval_pred):
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   f1_macro = f1_score(labels, predictions, average='macro')
   f1 = f1_score(labels, predictions, average=None)
   wandb.log({"f1": f1, "f1_macro": f1_macro})
   return {"f1": f1, "f1_macro": f1_macro}

if __name__ == "__main__":
    sys.path.append('../data')
    train_path = '../data/preprocessed/train.parquet'
    test_path = '../data/preprocessed/test.parquet'
    model_path = '../data/model'
    # check if model path exists
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train(train_data_path=train_path, test_data_path=test_path, model_path=model_path)