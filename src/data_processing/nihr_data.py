import pandas as pd


def read_nihr_dataset():
    """Reads NIHR dataset and clean columns.

    Returns:
        pd.DataFrame: Cleaned NIHR dataset.

    """
    nihr_df = pd.read_parquet(
        'data/raw/nihr_all.parquet',
        columns=[
            'project_id',
            'project_title',
            'scientific_abstract',
            'ukcrc_value',
            'hrcs_rac_category',
            'ukcrc_value_rac'
        ]
    )

    col_names = {
        'ukcrc_value': 'HC',
        'hrcs_rac_category': 'RA_top',
        'ukcrc_value_rac': 'RA',
        'project_title': 'AwardTitle',
        'scientific_abstract': 'AwardAbstract',
        'project_id': 'OrganisationReference'
    }
    clean_df = nihr_df.rename(columns=col_names)
    clean_df.dropna(subset=['RA_top', 'RA'], inplace=True)

    clean_df = clean_df.loc[
        ~clean_df['RA'].str.split().str[0].isin(['9.1', '9.2'])
    ]

    null_hcs = [
        'Awaiting Health Category Coding',
        'Will Not Be Health Category Coded'
    ]

    clean_df.where(clean_df['HC'].isin(null_hcs), None)
    clean_df['HC'] = clean_df['HC'].str.split(',')

    clean_df['RA_top'] = clean_df['RA_top'].str.strip().str.split('/')

    clean_df['RA_top'] = clean_df['RA_top'].apply(
        lambda list: [
            x.strip()[:1] for x in list if pd.notnull(x) and x != 'None'
        ]
    )

    clean_df['RA'] = clean_df['RA'].str.split('/')

    clean_df['RA'] = clean_df['RA'].apply(
        lambda lst: [s.split()[0] for s in lst]
    )

    clean_df['FundingOrganisation'] = 'Department of Health and Social Care'

    return clean_df


if __name__ == '__main__':
    read_nihr_dataset()
