import numpy as np
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

    df = df.sort_values(by='year')

    df.drop_duplicates(
        subset=[
            'FundingOrganisation',
            'OrganisationReference',
            'AwardTitle',
            'AwardAbstract',
        ],
        inplace=True,
        keep='last'
    )

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
    df['TextLen'] = df['AllText'].str.len()
    df = df.loc[df['TextLen'] >= 5]

    df.fillna('', inplace=True)
    df = df.astype(str)
    df.to_parquet('data/clean/ukhra_combined.parquet')

    return df


def collate_labels(df):
    """Collates selected labels into a list.

    Args:
        df(pd.DataFrame): UKHRA dataset.

    Returns:
        pd.DataFrame: Transformed UKHRA dataset.

    """
    id_cols = [
        'FundingOrganisation',
        'OrganisationReference',
        'AwardTitle',
        'AwardAbstract',
        'AllText',
        'TextLen',
        'year'
    ]

    label_list = ['RA', 'RA_top', 'HC']
    for label in label_list:
        cat_cols = [c for c in df if f'{label}_' in c and c[-1] != '%']

        # Lists to store category values
        cat_list = []

        # Iterate over each row in the dataframe
        for _, row in df.iterrows():
            # Create a temporary list for each row
            grant_cat = []

            # Iterate over RA_ columns
            for col in cat_cols:
                if len(row[col]) >= 3:  # Check if the value is not null
                    if label == 'RA':
                        grant_cat.append(row[col][:3])
                    elif label == 'HC':
                        hc_value = row[col].lower()
                        hc_value = hc_rename(hc_value)
                        grant_cat.append(hc_value)

            # Append the list of non-null `RA_` values to ra_list
            cat_list.append(grant_cat)

        df[label] = cat_list

    df = df[id_cols+label_list]
    # drop empy lists in the label column
    df = df.loc[df[label].str.len() > 0]
    df = df.reset_index()

    return df


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


def build_dataset():
    """Builds single cleaned dataset from downloaded files"""
    combined_df = combine_ukhra_datasets()
    process(combined_df)


if __name__ == '__main__':
    build_dataset()
