import numpy as np
import pandas as pd
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

    for term in (
        'Background',
        'Background,',
        'Background:',
        'BACKGROUND',
        'Objectives'
    ):
        # ToDo: Replace with regex.
        df['AwardAbstract'] = df['AwardAbstract'].str.replace(term, '')

    df['AwardAbstract'] = df['AwardAbstract'].str.strip()
    df['AllText'] = df['AwardTitle'] + ' ' + df['AwardAbstract']
    df.drop(columns=['AwardTitle', 'AwardAbstract'], inplace=True)
    df = df.loc[df['AllText'].str.len() >= 20].copy()

    df.drop_duplicates(subset='AllText', inplace=True, keep='last')

    return df


def build_dataset():
    """Builds single cleaned dataset from downloaded files"""
    print('Loading UKHRA datasets...')
    combined_ukhra_df = load_combined_ukhra_datasets()
    print('Loading NIHR datasets...')
    nihr_df = read_nihr_dataset()

    nihr = 'Department of Health and Social Care'
    nihr_refs = combined_ukhra_df.loc[
        combined_ukhra_df['FundingOrganisation'] == nihr
    ]['OrganisationReference']

    nihr_df = nihr_df.loc[nihr_df['OrganisationReference'].isin(nihr_refs)]

    print('Combining and cleaning datasets...')
    df = pd.concat([combined_ukhra_df, nihr_df])
    df = process_abstracts(df)

    # Mixed org data types cause a pyarrow error when saving to parquet.
    df['OrganisationReference'] = df['OrganisationReference'].astype(str)
    df.to_parquet('data/clean/clean.parquet', index=False)


if __name__ == '__main__':
    build_dataset()
