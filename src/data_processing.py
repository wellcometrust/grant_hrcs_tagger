import pandas as pd
from rich.progress import track


def read_ukhra_dataset(year):
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
    df = []
    for year in track([2014, 2018, 2022]):
        ukhra_dataset = read_ukhra_dataset(year)
        df.append(ukhra_dataset)

    df = pd.concat(df)

    df.drop_duplicates(
        subset=[
            'FundingOrganisation',
            'OrganisationReference',
            'AwardTitle',
            'AwardAbstract'
        ],
        inplace=True
    )

    df = df.astype(str)
    df.to_parquet('data/clean/ukhra_combined.parquet')

    return df


def melt_labels(df, label):
    id_cols = [
        'FundingOrganisation',
        'OrganisationReference',
        'AwardTitle',
        'AwardAbstract'
    ]

    cols = id_cols + [c for c in df if f'{label}_' in c and c[-1] != '%']
    df = df[cols].melt(id_vars=id_cols, ignore_index=True)
    df.rename(columns={'value': label}, inplace=True)
    df.dropna(subset=label, inplace=True)
    df[id_cols + [label]]

    return df


if __name__ == '__main__':
    combined_df = combine_ukhra_datasets()

    ra_df = melt_labels(combined_df, 'RA')
    ra_df.to_parquet('data/clean/ukhra_ra.parquet')

    hc_df = melt_labels(combined_df, 'HC')
    hc_df.to_parquet('data/clean/ukhra_hc.parquet')
