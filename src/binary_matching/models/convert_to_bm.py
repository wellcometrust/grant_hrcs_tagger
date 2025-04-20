import pandas as pd


def load_criteria():
    category_df = pd.read_csv('data/criteria/criteria.csv', header=None)
    category_df.columns = ['variable', 'criteria']
    category_df['variable'] = category_df['variable'].astype(str)

    return category_df


def transform_data(
    input_fp,
    output_fp,
    criteria,
    columns=['1.1', '2.1', '5.1', '4.1', '6.1', '8.1', '4.2', '7.1', 'text']
):
    df = pd.read_parquet(input_fp)
    df = df[columns]

    df = df.melt(id_vars='text')

    df = df.merge(criteria, how='left', on='variable')
    df = df.sample(frac=1).reset_index(drop=True)

    df['variable'] = df['variable'].astype(float)

    df.to_parquet(output_fp, index=False)


def convert_for_binary_matching():
    criteria_df = load_criteria()

    transform_data(
        'data/preprocessed/ra/train.parquet',
        'data/preprocessed/ra/train_match.parquet',
        criteria_df
    )
    
    transform_data(
        'data/preprocessed/ra/test.parquet',
        'data/preprocessed/ra/test_match.parquet',
        criteria_df
    )


if __name__ == '__main__':
    convert_for_binary_matching()
