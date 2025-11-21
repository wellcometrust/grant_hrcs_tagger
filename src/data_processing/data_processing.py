import akin
import json
import numpy as np
import pandas as pd
import string
from nihr_data import read_nihr_dataset
from ukhra_data import load_combined_ukhra_datasets

with open('src/data_processing/config/config.json', 'rt') as config_file:
    config = json.load(config_file)

with open('src/data_processing/config/hc_mapping.json', 'rt') as hc_map_file:
        hc_map = json.load(hc_map_file)


def hc_rename(hc_values):
    """Standardise HC naming across the dataset.

    Args:
        hc_value(str): health category name.

    Returns:
        list: Standardised health category names.

    """
    hc_values = hc_values.apply(lambda lst: [x.strip().lower() for x in lst])
    hc_values = hc_values.apply(lambda lst: [hc_map.get(x, x) for x in lst])

    return hc_values


def deduplicate(df):
    """Remove duplicate and near duplicate texts from AllText column.

    Args:
        df(pd.DataFrame): Dataframe containing text corpus to deduplicate.

    Returns:
        pd.DataFrame: Dataframe with duplicates removed.

    """
    # Remove exact duplicate text (case insensitive).
    df["lower"] = df["AllText"].str.lower()
    df.drop_duplicates(subset="lower", inplace=True, keep="last")

    # Remove near duplicate texts using locality sensitive hashing.
    minhash = akin.UniMinHash(seed=7)
    signatures = minhash.transform(df["lower"])
    lsh = akin.LSH(minhash.permutations, seed=1)
    lsh.update(signatures, df.index)
    adj = lsh.adjacency_list(min_jaccard=0.6)

    deduplicated_ids = set()
    duplicate_ids = set()
    for node, neighbours in adj.items():
        if node not in duplicate_ids:
            deduplicated_ids.add(node)
            duplicate_ids.update(neighbours)

    df = df.filter(list(deduplicated_ids), axis=0)
    df.drop(columns=["lower"], inplace=True)

    return df


def process_texts():
    """Clean and combine title and abstract texts.

    Args:
        df(pd.DataFrame): Dataset containing AwardTitle and AwardAbstract str columns.

    Returns:
        pd.DataFrame: Returns dataframe with cleaned and combined AllText str column.
    
    """



def process_texts(df):
    """Clean and combine title and abstract texts.

    Args:
        df(pd.DataFrame): Dataset containing AwardTitle and AwardAbstract str columns.

    Returns:
        pd.DataFrame: Returns dataframe with cleaned and combined AllText str column.
    
    """
    # Drop grants with titles and abstracts that are not informative
    df = df.loc[~df["AwardTitle"].str.strip().str.lower().isin(config["title_drop"])]
    df = df.loc[~df["AwardAbstract"].str.strip().str.lower().isin(config["abstract_drop"])]

    # Replace null titles and abstracts with empty strings
    df["AwardTitle"] = np.where(
        df["AwardTitle"].str.strip().str.lower().isin(config["title_nulls"]), "", df["AwardTitle"]
    )

    df["AwardAbstract"] = np.where(
        df["AwardAbstract"].str.strip().str.lower().isin(config["abstract_nulls"]),
        "",
        df["AwardAbstract"],
    )

    df["AwardTitle"] = df["AwardTitle"].fillna("")
    df = df.loc[~df["AwardTitle"].str.lower().str.startswith(config["title_prefixes"])]

    # Remove common funder specific boiler plate prefixes from abstracts.
    for term in config["funder_boiler_plate"]:
        df["AwardAbstract"] = np.where(
            df["AwardAbstract"].str[: len(term)].str.lower() == term,
            df["AwardAbstract"].str[len(term) + 1 :],
            df["AwardAbstract"]
        )

    removal_chars = string.punctuation + string.whitespace
    df["AwardAbstract"] = df["AwardAbstract"].str.lstrip(removal_chars)
    df["AwardAbstract"] = df["AwardAbstract"].str.strip()

    df["AllText"] = df["AwardTitle"].fillna("") + " " + df["AwardAbstract"].fillna("")
    df["AllText"] = df["AllText"].str.strip()
    df["AllText"] = df["AllText"].str.replace(r'\s+', ' ', regex=True)

    df = df.loc[df["AllText"].str.len() >= 110].copy()

    df = deduplicate(df)

    return df


def combine_datasets(ukhra_df, nihr_df):
    """Concatonates UKHRA and NIHR datasets into a single dataset.

    Removes duplicates based on NIHR organisational reference code.

    Args:
        ukhra_df: UKHRA dataset.
        nihr_df: NIHR dataset.

    Returns:
        pd.DataFrame: Combined dataset.

    """
    nihr_refs = ukhra_df.loc[
        ukhra_df["FundingOrganisation"] == "Department of Health and Social Care"
    ]["OrganisationReference"]

    nihr_df = nihr_df.loc[nihr_df["OrganisationReference"].isin(nihr_refs)]

    print("Combining and cleaning datasets...")
    df = pd.concat([ukhra_df, nihr_df], ignore_index=True)

    return df


def build_dataset():
    """Builds single cleaned dataset from downloaded files"""
    print("Loading UKHRA datasets...")
    combined_ukhra_df = load_combined_ukhra_datasets()
    print("Loading NIHR datasets...")
    nihr_df = read_nihr_dataset()

    df = combine_datasets(combined_ukhra_df, nihr_df)
    df["OrganisationReference"] = df["OrganisationReference"].astype(str)
    df["index"] = df.index

    df.to_parquet("data/clean/pre_clean.parquet", index=False)

    df = process_texts(df)
    df["HC"] = hc_rename(df["HC"])

    # Mixed org data types cause a pyarrow error when saving to parquet.
    df["OrganisationReference"] = df["OrganisationReference"].astype(str)
    df.to_parquet("data/clean/clean.parquet", index=False)


if __name__ == "__main__":
    build_dataset()
