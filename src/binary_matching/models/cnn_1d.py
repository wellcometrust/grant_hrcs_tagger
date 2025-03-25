import numpy as np
import pandas as pd
from rich.progress import track
from sklearn import metrics
import torch
from torch import nn


torch.manual_seed(10)


def init_device():
    """ Initialize device to use for training.

    Returns:
        str: device to use for training
    """
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_built():
        return "mps"
    raise SystemError("No GPU acceleration available.")


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


class HRCSDataset(torch.utils.data.Dataset):
    """
    A custom Dataset class for handling encodings and labels for training and
    evaluation.
    """
    def __init__(self, file_path):
        df = pd.read_parquet(file_path)
        self.embeddings = df['embeddings'].tolist()
        self.classes = [r for r in df[df.columns[:-2]].to_numpy()]

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        embedding = torch.as_tensor(self.embeddings[idx].tolist())
        embedding = embedding.unsqueeze(0)
        labels = torch.as_tensor(self.classes[idx].astype(np.float32).tolist())
        return embedding, labels


batch_size = 100

train_data = HRCSDataset('data/embeddings/train.parquet')
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

test_data = HRCSDataset('data/embeddings/test.parquet')
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size
)


model = ConvNet1D()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = init_device()
model.to(device)

print('Training model:')
for epoch in track(range(40)):
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_hat = model(X_batch)
        optimizer.zero_grad()
        loss = loss_fn(y_hat, y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}, loss: {loss.item()}')

print('Evaluating model:')
model.eval()
y_true = []
y_predicted = []
with torch.no_grad():
    for X_batch, y_batch in track(test_loader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_hat = model(X_batch).tolist()
        y_hat = torch.tensor(y_hat)
        y_hat = torch.where(y_hat > 0.5, 1, 0)

        y_true += y_batch.tolist()
        y_predicted += y_hat.tolist()

f1_macro = metrics.f1_score(y_true, y_predicted, average='macro')
f1_micro = metrics.f1_score(y_true, y_predicted, average='micro')
accuracy = metrics.accuracy_score(y_true, y_predicted)

print(f'Accuracy: {accuracy}')
print(f'F1 macro: {f1_macro}')
print(f'F1 micro: {f1_micro}')
