import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """

    quasi_ids = ['age', 'gender', 'zip3']
    
    merged = pd.merge(anon_df, aux_df, on=quasi_ids)

    counts = merged.groupby(quasi_ids).size().reset_index(name='count')
    unique_combinations = counts[counts['count'] == 1][quasi_ids]
    
    unique_matches = pd.merge(merged, unique_combinations, on=quasi_ids)

    if 'name' in unique_matches.columns:
        unique_matches = unique_matches.rename(columns={'name': 'matched_name'})
    elif 'Name' in unique_matches.columns:
        unique_matches = unique_matches.rename(columns={'Name': 'matched_name'})

    return unique_matches[['anon_id', 'matched_name']]


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
        
    return len(matches_df) / len(anon_df)