import torch
from torch import nn
from transformers import AutoModel


class BinaryMatchingMLP(torch.nn.Module):
    """A custom model class for binary classification using a pre-trained
    transformer model.
    """
    def __init__(self):
        super().__init__()
        self.cross_encoder = AutoModel.from_pretrained(
            'distilbert/distilbert-base-uncased'
        )

        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(768, 1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        x = self.cross_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        x = x.last_hidden_state[:, 0, :]  # Get hidden CLS token
        x = self.dropout(x)
        x = self.activation(self.linear(x))

        return x


class ConvNet1D(nn.Module):
    def __init__(self, n_filters=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=5, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.mlp1 = nn.Linear(24512, 1000)
        self.act2 = nn.ReLU()

        self.dropout = nn.Dropout(0.1)

        self.mlp4 = nn.Linear(1000, 48)
        self.act5 = nn.Sigmoid()

    def forward(self, x):
        # Feature extraction ConvNet
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.flatten(x)

        # Classification MLP
        x = self.act2(self.mlp1(x))
        x = self.dropout(x)

        # Sigmoid activation for multilabel classification
        x = self.act5(self.mlp4(x))

        return x
