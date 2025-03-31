import os

from sklearn import metrics
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from tqdm import tqdm

from data import get_train_dataloader, get_test_dataloader
from model import BinaryMatchingMLP


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
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    init_process_group('nccl', rank=rank, world_size=world_size)


def train(model, train_loader, rank, optimizer, loss_fn):
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
    model.eval()
    y_true = []
    y_predicted = []
    with torch.no_grad():
        for batch in tqdm(test_loader, disable=not rank == 0, colour='blue'):
            X_batch = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            y_batch = batch['labels'].to(rank)

            y_hat = model(X_batch, attention_mask).tolist()
            y_hat = torch.tensor(y_hat)
            y_hat = torch.where(y_hat > 0.5, 1, 0)

        y_true += y_batch.tolist()
        y_predicted += y_hat.tolist()

    f1_macro = metrics.f1_score(y_true, y_predicted, average='macro')
    accuracy = metrics.accuracy_score(y_true, y_predicted)

    f1_macro = torch.tensor(f1_macro).to(rank)
    accuracy = torch.tensor(accuracy).to(rank)

    return f1_macro, accuracy


def worker(
    rank,
    world_size,
    model,
    n_epochs,
    batch_size,
    learning_rate=0.00001
):
    if world_size:
        setup_distributed_training(rank, world_size)

    model = model.to(rank)

    if world_size:
        model = DDP(model, device_ids=[rank])

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if world_size:
        train_loader = get_train_dataloader(batch_size, distributed=world_size)

    n_steps = len(train_loader)

    for epoch in range(n_epochs):
        # ToDo: Add shuffling.

        train_loader = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1} of {n_epochs}',
            total=n_steps,
            disable=not rank == 0,
            colour='green'
        )

        loss = train(model, train_loader, rank, optimizer, loss_fn)

        torch.distributed.barrier()
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)

        if rank == 0:
            print(f'Loss: {loss}')

    if world_size:
        test_loader = get_test_dataloader(batch_size, distributed=world_size)

    f1_macro, accuracy = evaluate(model, test_loader, rank)

    torch.distributed.barrier()
    torch.distributed.all_reduce(f1_macro, op=torch.distributed.ReduceOp.AVG)
    torch.distributed.all_reduce(accuracy, op=torch.distributed.ReduceOp.AVG)

    if rank == 0:
        print(f'Accuracy: {accuracy}')
        print(f'F1 Macro: {f1_macro}')

    if world_size:
        destroy_process_group()


def main(model, multi_gpu=True, n_epochs=2, batch_size=100):
    print('Training model:')

    torch.cuda.empty_cache()
    device = get_device()

    world_size = torch.cuda.device_count()

    if world_size > 1 and multi_gpu and device == 'cuda':
        print(f'Using CUDA to train on {world_size} GPUs')
        mp.spawn(
            worker,
            args=(world_size, model, n_epochs, batch_size),
            nprocs=world_size,
            join=True
        )
    else:
        print(f'Using {device.upper()} to train on a single GPU')
        worker(0, None, model, n_epochs, batch_size)


if __name__ == '__main__':
    model = BinaryMatchingMLP()
    main(model, n_epochs=10)
