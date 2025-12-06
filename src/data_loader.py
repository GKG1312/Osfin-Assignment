import pandas as pd
import os

def load_data(data_dir="data"):
    """
    Loads disputes and transactions data from CSV files.
    """
    disputes_path = os.path.join(data_dir, "disputes.csv")
    transactions_path = os.path.join(data_dir, "transactions.csv")
    
    try:
        disputes = pd.read_csv(disputes_path)
        transactions = pd.read_csv(transactions_path)
        return disputes, transactions
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def merge_data(disputes, transactions):
    """
    Merges disputes with transactions on txn_id.
    """
    # Merge, keeping all disputes
    merged = pd.merge(disputes, transactions, on="txn_id", how="left", suffixes=("_dispute", "_txn"))
    return merged
