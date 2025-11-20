import akin
import numpy as np
import pandas as pd
import string
from nihr_data import read_nihr_dataset
from ukhra_data import load_combined_ukhra_datasets


def hc_rename(hc_values):
    """Steamline HC naming

    Args:
        hc_value(str): health category name

    Return:
        str: renamed health category name

    """
    hc_values = hc_values.apply(lambda lst: [x.strip().lower() for x in lst])

    streamline_dict = {
        "cancer": "cancer and neoplasms",
        "cardio": "cardiovascular",
        "congenital": "congenital disorders",
        "inflammatory": "inflammatory and immune system",
        "inflamation and immune": "inflammatory and immune system",
        "inflammatory and immune system": "inflammatory and immune system",
        "injuries": "injuries and accidents",
        "mental": "mental health",
        "metabolic": "metabolic and endocrine",
        "muscle": "musculoskeletal",
        "oral": "oral and gastrointestinal",
        "renal": "renal and urogenital",
        "reproduction": "reproductive health and childbirth",
        "generic": "generic health relevance",
        "other": "disputed aetiology and other",
    }

    hc_values = hc_values.apply(lambda lst: [streamline_dict.get(x, x) for x in lst])

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


def process_abstracts(df):
    """Clean and combine title and abstract texts."""
    # titles and abstracts for which we like to drop the grants
    title_drop = [
        "mrc studentship - award title not available in public dataset",
        "award title unavailable in public datasetredacted for public dataset",
        "no data entered",
    ]

    abstract_drop = [
        "award abstract not available in public dataset",
        "award abstract unavailable in public dataset"
        "mrc studentship - award abstract not available in public dataset",
        "redacted for public dataset",
    ]

    # titles and abstracts for grants we like to keep but for which the title or abstract is not informative
    title_nulls = [
        "award title unavailable",
        "award title unavailable in public dataset",
        "no title available",
    ]

    abstract_nulls = [
        "(pivotal) study",
        "abstract not available",
        "award abstract not available in public dataset",
        "award abstract unavailable for analysis or public dataset",
        "awardabstract",
        "no data entered",
        "cso nrs career research fellowship award - no abstract available",
        "nihr biomedical research centre (brc) award - no abstract available",
        "nihr biomedical research unit (bru) award - no abstract available",
        "nihr collaboration for leadership in applied health research and care (clahrc) award - no abstract available",
        "nihr healthcare technology cooperative (htc) - no abstract available",
        "nihr imperial patient safety translational research centre - no abstract available",
        "no abstract",
        "no abstract available for this analysis",
        "no abstract available for this analysis.",
        "no abstract available/provided",
        "no abstract available/provided or marked confidential",
        "no abstract provided for this analysis",
        "paper abstract only",
        "rare diseases translational research collaboration (trc) - no abstract available",
    ]

    # Drop grants with titles and abstracts that are not informative
    df = df.loc[~df["AwardTitle"].str.strip().str.lower().isin(title_drop)]
    df = df.loc[~df["AwardAbstract"].str.strip().str.lower().isin(abstract_drop)]

    # Replace null titles and abstracts with empty strings
    df["AwardTitle"] = np.where(
        df["AwardTitle"].str.strip().str.lower().isin(title_nulls), "", df["AwardTitle"]
    )

    df["AwardAbstract"] = np.where(
        df["AwardAbstract"].str.strip().str.lower().isin(abstract_nulls),
        "",
        df["AwardAbstract"],
    )

    df["AwardTitle"] = df["AwardTitle"].fillna("")

    prefixes = (
        "pivotal nurse support contract",
        "nihr in-practice fellowship",
        "nurture: national unified renal",
        "qi project: assist-ckd"
    )

    df = df.loc[~df["AwardTitle"].str.lower().str.startswith(prefixes)]

    # Remove common funder specific boiler plate prefixes from abstracts.
    for term in (
        "background",
        "aim",
        "aims",
        "background",
        "objectives",
        "objective",
        "mica",
        "what is the project about",
        "design",
        "description"
    ):
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


def build_dataset():
    """Builds single cleaned dataset from downloaded files"""
    print("Loading UKHRA datasets...")
    combined_ukhra_df = load_combined_ukhra_datasets()
    print("Loading NIHR datasets...")
    nihr_df = read_nihr_dataset()

    nihr = "Department of Health and Social Care"
    nihr_refs = combined_ukhra_df.loc[
        combined_ukhra_df["FundingOrganisation"] == nihr
    ]["OrganisationReference"]

    nihr_df = nihr_df.loc[nihr_df["OrganisationReference"].isin(nihr_refs)]

    print("Combining and cleaning datasets...")
    df = pd.concat([combined_ukhra_df, nihr_df], ignore_index=True)

    df["OrganisationReference"] = df["OrganisationReference"].astype(str)
    df["index"] = df.index

    df.to_parquet("data/clean/pre_clean.parquet", index=False)

    df = process_abstracts(df)
    df["HC"] = hc_rename(df["HC"])

    # Mixed org data types cause a pyarrow error when saving to parquet.
    df["OrganisationReference"] = df["OrganisationReference"].astype(str)
    df.to_parquet("data/clean/clean.parquet", index=False)


if __name__ == "__main__":
    build_dataset()
