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
    else:
        ukhra_df = ukhra_df.loc[ukhra_df['CodingType'] == 'Manual']

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


def unpivot_ra_labels(df, prefix):
    unpivot_cols = [col for col in list(df) if col.split('_')[0] == prefix]

    df[prefix] = df.apply(
        lambda row: [row[col] for col in unpivot_cols if pd.notnull(row[col])],
        axis=1
    )

    return df


def load_combined_ukhra_datasets():
    """Read, clean and combine UKHRA datasets into a single dataframe.

    Returns:
        pd.DataFrame: Combined UKHRA dataset.

    """
    df = []
    for year in track([2014, 2018, 2022]):
        ukhra_dataset = read_ukhra_dataset(year)
        df.append(ukhra_dataset)

    df = pd.concat(df)
    df = df.sort_values(by='year')
    df.drop_duplicates(subset=list(df)[:2], inplace=True, keep='last')
    df.reset_index(drop=True, inplace=True)

    df = unpivot_ra_labels(df, 'RA')
    df = unpivot_ra_labels(df, 'HC')

    df['RA_top'] = df['RA'].apply(lambda x: list(set([str(ra)[0] for ra in x])))
    df = df[list(df)[:4] + ['RA', 'RA_top', 'HC']]

    return df


load_combined_ukhra_datasets()
