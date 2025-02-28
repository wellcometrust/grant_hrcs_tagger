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


def unpivot_labels(df, prefix):
    """Unpivot label columns.

    Args:
        df(pd.DataFrame): Dataframe containing columns to unpivot.
        prefix(str): Prefix of columns to unpivot.

    Returns:
        pd.DataFrame: Dataframe containing unpivotted columns.

    """
    unpivot_cols = [col for col in list(df) if col.split('_')[0] == prefix]

    df[prefix] = df[unpivot_cols].apply(
        lambda row: [x for x in row if pd.notnull(x)],
        axis=1
    )

    df.drop(columns=unpivot_cols, inplace=True)

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

    df = unpivot_labels(df, 'HC')
    df = unpivot_labels(df, 'RA')
    df['RA'] = df['RA'].apply(
        lambda list: [str(x)[:3] for x in list if str(x).strip()]
    )

    df['RA_top'] = df['RA'].apply(
        lambda x: list(set([ra[0] for ra in x if ra.strip()]))
    )

    df = df[list(df)[:4] + ['RA', 'RA_top', 'HC']]

    return df


if __name__ == '__main__':
    load_combined_ukhra_datasets()
