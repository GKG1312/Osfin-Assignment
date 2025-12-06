import os
import warnings

from src.data_loader import load_data, merge_data
from src.classifier import process_classification
from src.resolver import process_resolutions


def main():
    print("Loading data...")
    disputes, transactions = load_data()
    
    if disputes is None or transactions is None:
        print("Data load failed.")
        return

    print("Task 1: Classifying Disputes...")
    classified_df = process_classification(disputes)
    
    # Save Output 1
    output1_path = "classified_disputes.csv"
    classified_df.to_csv(output1_path, index=False)
    print(f"Saved {output1_path}")
    
    print("Task 2: Suggesting Resolutions...")
    # We need to merge classification back to disputes to get txn_id, then merge with transactions
    # But process_classification returned only the new cols + dispute_id
    
    # Join classified back to original disputes
    classified_full = disputes.merge(classified_df, on="dispute_id")
    
    # Merge with transactions to get status, timestamp, etc
    merged_data = merge_data(classified_full, transactions)
    
    # Generate resolutions
    resolutions_df = process_resolutions(merged_data, transactions)
    
    # Save Output 2
    output2_path = "resolutions.csv"
    resolutions_df.to_csv(output2_path, index=False)
    print(f"Saved {output2_path}")
    
    print("Processing Complete.")

if __name__ == "__main__":
    main()
