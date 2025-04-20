import os

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from tqdm import tqdm

from data import get_train_dataloader, get_test_dataloader
from model import BinaryMatchingMLP
import metrics

torch.manual_seed(10)


def get_device():
    """ Initialize device to use for training.

    Returns:
        str: device to use for training

    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_built():
        return "mps"

    raise SystemError("No GPU acceleration available.")


def setup_distributed_training(rank, world_size):
    """Set up process group.

    Args:
        rank(int): GPU device ID.
        world_size(int): Total number of GPUs.

    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11111'

    init_process_group('nccl', rank=rank, world_size=world_size)


def train(model, train_loader, rank, optimizer, loss_fn):
    """ Performs a training epoch.

    Args:
        model: Transformers AutoModel class.
        train_loader: PyTorch DistributedSampler module.
        rank(int): GPU device id.
        optimizer: Optimiser module for training.
        loss_fn: Loss function module for training.

    Returns:
        int: Loss value.

    """
    for batch in train_loader:
        X_batch = batch['input_ids'].to(rank)
        attention_mask = batch['attention_mask'].to(rank)
        y_batch = batch['labels'].to(rank)

        y_hat = model(X_batch, attention_mask)
        y_hat = y_hat.squeeze(1)

        optimizer.zero_grad()
        loss = loss_fn(y_hat, y_batch)
        loss.backward()
        optimizer.step()

    return loss


def evaluate(model, test_loader, rank):
    """ Evaluate model agaisnt hold out test dataset.

    Args:
        model: Fine-tuned Transformers AutoModel class.
        test_loader: PyTorch DistributedSampler module.
        rank(int): GPU device id.

    Returns:
        tuple: Model performance metrics.

    """
    model.eval()
    y_true = []
    y_predicted = []
    class_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, disable=rank != 0, colour='blue'):
            X_batch = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            y_batch = batch['labels'].to(rank)

            class_labels += batch['classes']
            y_true.append(y_batch)

            y_hat = model(X_batch, attention_mask).tolist()
            y_hat = torch.tensor(y_hat, device=rank)
            y_hat = torch.where(y_hat > 0.5, 1, 0)
            y_predicted.append(y_hat.squeeze())

    y_true = torch.cat(y_true, dim=0)
    y_predicted = torch.cat(y_predicted, dim=0)
    class_labels = torch.tensor(class_labels, device=rank).squeeze()
    eval_data = torch.stack([y_true, y_predicted, class_labels])

    return torch.transpose(eval_data, 0, 1)


def worker(
    rank,
    world_size,
    model,
    n_epochs,
    batch_size,
    queue,
    learning_rate=0.00001
):
    """ Distributed function for co-ordinating experiments on a GPU worker.

    Args:
        rank(int): GPU device ID.
        world_size(int): Total number of GPUs.
        model: Fine-tuned Transformers AutoModel class.
        n_epochs(int): Number of training epochs.
        batch_size(int): Batch chunk size.
        learning_rate(float): Learning weight for training.
    
    """
    if world_size:
        setup_distributed_training(rank, world_size)

    model = model.to(rank)

    if world_size:
        model = DDP(model, device_ids=[rank])

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = get_train_dataloader(batch_size, distributed=world_size)

    n_steps = len(train_loader)

    for epoch in range(n_epochs):
        # ToDo: Add shuffling.

        train_loader = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1} of {n_epochs}',
            total=n_steps,
            disable=rank != 0,
            colour='green'
        )

        loss = train(model, train_loader, rank, optimizer, loss_fn)

        if world_size:
            torch.distributed.barrier()
            torch.distributed.reduce(
                loss,
                dst=0,
                op=torch.distributed.ReduceOp.AVG,
            )

        if rank == 0:
            print(f'Loss: {loss}')

    test_loader = get_test_dataloader(batch_size, distributed=world_size)
    predictions = evaluate(model, test_loader, rank)

    predictions = predictions.detach().cpu()
    predictions = predictions.numpy()
    queue.put(predictions)

    if world_size:
        destroy_process_group()


def main(model, multi_gpu=True, n_epochs=2, batch_size=100):
    """ Co-ordinates training and evaluation across multiple GPUs.

    Args:
        model: Fine-tuned Transformers AutoModel class.
        multi_gpu(bool): Use multiple GPUs for training if available.
        n_epochs(int): Number of training epochs.
        batch_size(int): Batch chunk size.

    """
    print('Training model:')

    torch.cuda.empty_cache()
    device = get_device()

    world_size = torch.cuda.device_count()
    if world_size > 1 and multi_gpu and device == 'cuda':
        print(f'Using CUDA to train on {world_size} GPUs')
        mp.set_start_method("spawn")
        queue = mp.Queue()

        mp.spawn(
            worker,
            args=(world_size, model, n_epochs, batch_size, queue),
            nprocs=world_size,
            join=False
        )

        eval_data = []
        while True:
            eval_data.append(queue.get())
            if len(eval_data) == world_size:
                break
    
        eval_data = np.concatenate(eval_data)
        metrics.calculate_metrics(eval_data)

    else:
        print(f'Using {device.upper()} to train on a single GPU')
        worker(0, None, model, n_epochs, batch_size)


if __name__ == '__main__':
    model = BinaryMatchingMLP()
    main(model, n_epochs=2, multi_gpu=True)
