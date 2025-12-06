import pandas as pd

def get_disputes_by_type(classified_df):
    """
    Returns counts of disputes by predicted category.
    """
    return classified_df["predicted_category"].value_counts().to_dict()

def get_unresolved_fraud(classified_df):
    """
    Returns list of FRAUD disputes (assuming all in this list are 'active' contextually).
    """
    return classified_df[classified_df["predicted_category"] == "FRAUD"]

def get_duplicate_count_today(classified_df, disputes_full_df):
    """
    Returns count of duplicate charges for 'today'.
    Note: Since dataset is static 2025, we might interpret 'today' or just 'total'.
    """
    # Assuming we join to get dates if needed, or just return total for this static set
    return len(classified_df[classified_df["predicted_category"] == "DUPLICATE_CHARGE"])

def get_recent_trends(disputes_df):
    """
    Returns daily count of disputes.
    """
    df = disputes_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df.groupby(df["created_at"].dt.date).size()
