import pandas as pd
import torch
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


class Dataset(torch.utils.data.Dataset):
    """ Handles custom tokenisation and labels for training and evaluation.
    """
    def __init__(self, file_path, criteria):
        df = pd.read_parquet(file_path)
        self.corpus = df['text'].tolist()
        self.classes = df['1.1'].to_list()
        criteria_df = pd.read_csv(criteria, header=None)
        self.criteria = criteria_df.iloc[:, 1].to_list()

        self.tokenizer = AutoTokenizer.from_pretrained(
            'distilbert/distilbert-base-uncased'
        )

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        criteria = self.criteria[0]
        tokens = self.tokenizer.encode_plus(
            self.corpus[idx],
            text_pair=criteria,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt',
        )

        data_sample = {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.classes[idx], dtype=torch.float32)
        }

        return data_sample


def get_train_dataloader(batch_size=32, distributed=False, shuffle=True):
    """Get the training dataloader."""
    train_dataset = Dataset(
        'data/preprocessed/ra/train.parquet',
        'data/criteria/criteria.csv'
    )

    if distributed:
        shuffle = False
        sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler
    )

    return train_loader


def get_test_dataloader(batch_size=32, distributed=False):
    """Get the test dataloader."""
    test_dataset = Dataset(
        'data/preprocessed/ra/test.parquet',
        'data/criteria/criteria.csv'
    )

    if distributed:
        sampler = DistributedSampler(test_dataset, shuffle=True)
    else:
        sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    return test_loader
