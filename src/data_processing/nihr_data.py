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

    clean_df = nihr_df.loc[
        ~nihr_df['ukcrc_value_rac'].str.split().str[0].isin(['9.1', '9.2'])
    ]

    clean_df = clean_df.loc[clean_df['hrcs_rac_category'].str.split().str[0] != '9.']
    clean_df = clean_df.loc[clean_df['hrcs_rac_category'] != 'None']
    clean_df = clean_df.loc[clean_df['hrcs_rac_category'].str.split().str[0] != 'None']

    null_hcs = [
        'Awaiting Health Category Coding',
        'Will Not Be Health Category Coded'
    ]

    clean_df.where(clean_df['ukcrc_value'].isin(null_hcs), None)

    clean_df['ukcrc_value'] = clean_df['ukcrc_value'].str.split(',')
    clean_df['hrcs_rac_category'] = clean_df['hrcs_rac_category'].str.split('/')
    clean_df['ukcrc_value_rac'] = clean_df['ukcrc_value_rac'].str.split('/')

    clean_df = clean_df.dropna(subset=['hrcs_rac_category', 'ukcrc_value_rac'])

    clean_df['hrcs_rac_category'] = clean_df['hrcs_rac_category'].apply(
        lambda lst: [s.split()[0][:-1] for s in lst]
    )

    clean_df['ukcrc_value_rac'] = clean_df['ukcrc_value_rac'].apply(
        lambda lst: [s.split()[0] for s in lst]
    )

    col_names = {
        'ukcrc_value': 'HC',
        'hrcs_rac_category': 'RA_top',
        'ukcrc_value_rac': 'RA',
        'project_title': 'AwardTitle',
        'scientific_abstract': 'AwardAbstract',
        'project_id': 'OrganisationReference'
    }

    clean_df['FundingOrganisation'] = 'National Institute for Health Research'

    clean_df.rename(columns=col_names, inplace=True)

    return clean_df
