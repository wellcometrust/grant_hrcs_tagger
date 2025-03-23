import click
import pathlib
import pandas as pd
from models.embedding_model import EmbeddingModel


@click.group()
def cli():
    pass


@cli.command()
@click.option('--input_path', default='data/criteria/criteria.csv')
@click.option('--model', default='allenai/scibert_scivocab_uncased')
@click.option('--tokenizer', default='allenai/scibert_scivocab_uncased')
def process_criteria(
    input_path='data/criteria/criteria.csv',
    model='allenai/scibert_scivocab_uncased',
    tokenizer='allenai/scibert_scivocab_uncased'
):
    df = pd.read_csv(input_path, header=None)

    embedding_model = EmbeddingModel(
        tokenizer_name=tokenizer,
        model_name=model,
        parallelise=False
    )

    embeddings = embedding_model.run_inference(df[1].to_list())
    print(len(embeddings[0]))
    df['embeddings'] = embeddings

    df.to_parquet('data/embeddings/criteria.parquet', index=False)


@cli.command()
@click.option('--input_dir', default='data/preprocessed/ra')
@click.option('--model', default='allenai/scibert_scivocab_uncased')
@click.option('--tokenizer', default='allenai/scibert_scivocab_uncased')
def process_corpus(
    input_dir='data/preprocessed/ra',
    model='allenai/scibert_scivocab_uncased',
    tokenizer='allenai/scibert_scivocab_uncased'
):
    embedding_model = EmbeddingModel(
        tokenizer_name=tokenizer,
        model_name=model,
        parallelise=True
    )

    embedding_model.launch_processes()

    try:
        for file_type in ['train', 'test']:
            file_path = pathlib.PurePath(input_dir, f'{file_type}.parquet')
            df = pd.read_parquet(file_path)

            embeddings = embedding_model.run_inference(
                df['text'].to_list(),
                batch_size=100
            )

            df['embeddings'] = embeddings
            print(df.head())
            df.to_parquet(f'data/embeddings/{file_type}.parquet', index=False)

    finally:
        embedding_model.kill_processes()


if __name__ == "__main__":
    cli()
