import pandas as pd
from rich.progress import track


def read_ukhra_dataset(year):
    """Read UKHRA xlsx dataset and clean columns.

    Args:
        year(int): Year of UKHRA study.

    Returns:
        pd.DataFrame: UKHRA dataframe for given year.

    """
    ukhra_df = pd.read_excel(
        f'data/raw/ukhra{year}.xlsx',
        sheet_name=f'{year}_Data_1Line'
    )

    if year == 2014:
        col_map = {
            'GrantCode_Public': 'OrganisationReference',
            'Title_Public': 'AwardTitle',
            'Abstract_Public': 'AwardAbstract'
        }

        ukhra_df.rename(columns=col_map, inplace=True)

    cols = [
        'FundingOrganisation',
        'OrganisationReference',
        'AwardTitle',
        'AwardAbstract'
    ]

    cols += [c for c in ukhra_df if 'RA_' in c and c[-1] != '%']
    cols += [c for c in ukhra_df if 'HC_' in c and c[-1] != '%']
    ukhra_df = ukhra_df[cols]
    ukhra_df['year'] = year

    return ukhra_df


def combine_ukhra_datasets():
    """Read, clean and combine UKHRA datasets into a single file.

    Returns:
        pd.DataFrame: Combined UKHRA dataset.

    """
    df = []
    for year in track([2014, 2018, 2022]):
        ukhra_dataset = read_ukhra_dataset(year)
        df.append(ukhra_dataset)

    df = pd.concat(df)
    df = df.fillna('').astype(str)
    df.to_parquet('data/clean/ukhra_combined.parquet')

    return df


def melt_labels(df, label):
    """Converts selected label columns to rows.

    Args:
        df(pd.DataFrame): UKHRA dataset.
        label(str): Label column name.

    Returns:
        pd.DataFrame: Transformed UKHRA dataset.

    """
    id_cols = [
        'FundingOrganisation',
        'OrganisationReference',
        'AwardTitle',
        'AwardAbstract'
    ]

    cols = id_cols + [c for c in df if f'{label}_' in c and c[-1] != '%']
    df = df[cols].melt(id_vars=id_cols, ignore_index=True)
    df.rename(columns={'value': label}, inplace=True)
    df = df.loc[df[label] != '']
    df = df[id_cols + [label]]

    return df


def process_ra(df):
    """Transform and clean HRCS Research Activities.

    Args:
        df(pd.DataFrame): UKHRA dataset.

    """
    df = melt_labels(df, 'RA')
    df['RA1'] = df['RA'].str[0]
    df['RA2'] = df['RA'].str[:3]

    df.to_parquet('data/clean/ukhra_ra.parquet')


def process_hc(df):
    """Transform and clean HRCS Health Categories.

    Args:
        df(pd.DataFrame): UKHRA dataset.

    """
    df = melt_labels(df, 'HC')
    df['HC'] = df['HC'].str.lower()

    df['HC'].replace(
        {
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
        },
        inplace=True
    )

    df.to_parquet('data/clean/ukhra_hc.parquet')


if __name__ == '__main__':
    combined_df = combine_ukhra_datasets()
    process_ra(combined_df)
    process_hc(combined_df)
