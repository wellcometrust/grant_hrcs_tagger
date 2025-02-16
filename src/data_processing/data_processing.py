import numpy as np
import pandas as pd
from rich.progress import track
from nihr_data import read_nihr_dataset
from ukhra_data import load_combined_ukhra_datasets


def hc_rename(hc_value):
    """ Steamline HC naming

    Args:
        hc_value(str): health category name

    Return:
        str: renamed health category name
    """
    hc_streamline_dict = {
            'cancer': 'cancer and neoplasms',
            'cardio': 'cardiovascular',
            'congenital': 'congenital disorders',
            'inflammatory': 'inflammatory and immune system',
            'inflamation and immune': 'inflammatory and immune system',
            'inflammatory and immune system': 'inflammatory and immune system',
            'injuries': 'injuries and accidents',
            'mental': 'mental health',
            'metabolic': 'metabolic and endocrine',
            'muscle': 'musculoskeletal',
            'oral': 'oral and gastrointestinal',
            'renal': 'renal and urogenital',
            'reproduction': 'reproductive health and childbirth',
            'generic': 'generic health relevance',
            'other': 'disputed aetiology and other'
        }
    if hc_value in hc_streamline_dict:
        return hc_streamline_dict[hc_value]
    else:
        return hc_value


def process(df):
    """Transform and clean HRCS Research Activities.

    Args:
        df(pd.DataFrame): UKHRA dataset.

    """
    df = collate_labels(df)
    RA_top = []
    for _, row in df.iterrows():
        RA_top_row = []
        for ra in row['RA']:
            RA_top_row.append(ra[0])
        RA_top.append(RA_top_row)

    df['RA_top'] = RA_top

    df.to_parquet('data/clean/ukhra_clean.parquet')


def process_abstracts(df):
    """ Clean and combine title and abstract texts.

    """
    title_nulls = [
        'no title available',
        'award title not available in public dataset'
    ]

    abstract_nulls = [
        'award abstract unavailable in public dataset',
        'nihr collaboration for leadership in applied health research and care'
        ' (clahrc) award - no abstract available',
        'cso nrs career research fellowship award - no abstract available',
        'nihr biomedical research centre (brc) award - no abstract available',
        'nihr healthcare technology cooperative (htc) - no abstract available',
        'nihr imperial patient safety translational research centre -'
        ' no abstract available',
        'rare diseases translational research collaboration (trc)'
        ' - no abstract available',
        'no abstract'
    ]

    df['AwardTitle'] = np.where(
        df['AwardTitle'].str.strip().str.lower().isin(title_nulls),
        '',
        df['AwardTitle']
    )

    df['AwardAbstract'] = np.where(
        df['AwardAbstract'].str.strip().str.lower().isin(abstract_nulls),
        '',
        df['AwardAbstract']
    )

    df['AllText'] = df['AwardTitle'] + ' ' + df['AwardAbstract']


def build_dataset():
    """Builds single cleaned dataset from downloaded files"""
    combined_ukhra_df = load_combined_ukhra_datasets()
    nihr_df = read_nihr_dataset()
    df = pd.concat([combined_ukhra_df, nihr_df])
    print(df)
    #process(combined_df)


if __name__ == '__main__':
    build_dataset()

    #df['AllText'] = df['AwardTitle'] + ' ' + df['AwardAbstract']
    #df['TextLen'] = df['AllText'].str.len()
    #df = df.loc[df['TextLen'] >= 20]

    #df.drop_duplicates(subset='AllText', inplace=True, keep='last')

    #df.fillna('', inplace=True)
    #df = df.astype(str)