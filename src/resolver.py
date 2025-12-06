import pandas as pd
from datetime import timedelta

def check_potential_duplicates(current_txn, all_txns):
    """
    Checks if there is another SUCCESS transaction for the same user and amount
    within a short time window.
    """
    # Handle suffixed columns from merge
    cust_id = current_txn.get("customer_id")
    if pd.isna(cust_id):
        cust_id = current_txn.get("customer_id_dispute")
    if pd.isna(cust_id):
        cust_id = current_txn.get("customer_id_txn")
        
    amount = current_txn.get("amount")
    if pd.isna(amount):
        amount = current_txn.get("amount_dispute")
    if pd.isna(amount):
        amount = current_txn.get("amount_txn")
        
    current_txn_id = current_txn.get("txn_id")
    
    # Timestamp might be timestamp (from txn) or created_at (from dispute) logic?
    # Usually we compare transaction times.
    # The merged DF has 'timestamp' from txn (which might be 'timestamp' or 'timestamp_txn')
    current_time = current_txn.get("timestamp") # if no collision
    if pd.isna(current_time):
        current_time = current_txn.get("timestamp_txn")
        
    if pd.isna(current_time):
        # If we don't have the original transaction time, we can't do a time-based duplicate check easily
        # relative to the transaction. But we can check relative to dispute creation? No, dup check is txn vs txn.
        return False

    # Filter for same user, same amount, SUCCESS status
    candidates = all_txns[
        (all_txns["customer_id"] == cust_id) &
        (all_txns["amount"] == amount) &
        (all_txns["status"] == "SUCCESS") &
        (all_txns["txn_id"] != current_txn_id)
    ]
    
    if pd.isna(current_time):
        return False
        
    for _, candle in candidates.iterrows():
        # candle is from all_txns, so it has direct 'timestamp'
        time_diff = abs((candle["timestamp"] - current_time).total_seconds())
        # Let's say duplicate is within same day or short window. 
        # Assignment says "minutes apart" in examples. Let's use 60 mins to be safe.
        if time_diff <= 3600: 
            return True
            
    return False

def suggest_resolution(row, all_txns):
    """
    Suggests a resolution based on category and transaction details.
    """
    category = row.get("predicted_category")
    
    status = row.get("status")
    if pd.isna(status):
        status = row.get("status_txn")
    
    # Logic for DUPLICATE_CHARGE
    if category == "DUPLICATE_CHARGE":
        is_dup_confirmed = check_potential_duplicates(row, all_txns)
        if is_dup_confirmed:
            return "Auto-refund", "Found a duplicate successful transaction within short timeframe."
        else:
            return "Manual review", "User claims duplicate, but no matching success transaction found in near timeframe."

    # Logic for FRAUD
    if category == "FRAUD":
        return "Mark as potential fraud", "High severity claim. Immediate block and investigation required."

    # Logic for FAILED_TRANSACTION
    if category == "FAILED_TRANSACTION":
        if status == "FAILED":
            return "Auto-refund", "Transaction is marked FAILED in system but user reports debit. Refund."
        elif status == "SUCCESS":
            return "Ask for more info", "System shows SUCCESS but user claims failure. Need bank reference number."
        elif status == "PENDING":
             return "Check with Bank", "Transaction is PENDING. Check upstream status."
        else:
            return "Manual review", f"Unusual status: {status}"

    # Logic for REFUND_PENDING
    if category == "REFUND_PENDING":
        if status == "CANCELLED":
            return "Auto-refund", "Order was cancelled. Process refund if not done."
        elif status == "SUCCESS":
            return "Manual review", "User is waiting for refund on a SUCCESS transaction (possible return?)"
        else:
            return "Escalate to bank", "Refund delayed. Escalate."

    return "Manual review", "Category OTHERS or unclear rules."

def process_resolutions(merged_df, all_txns):
    """
    Generates resolutions for the merged dataframe.
    """
    # Ensure timestamps are datetime
    if "timestamp" in all_txns.columns:
        all_txns["timestamp"] = pd.to_datetime(all_txns["timestamp"])
    if "timestamp" in merged_df.columns:
        merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"])
    
    resolutions = []
    justifications = []
    
    for _, row in merged_df.iterrows():
        action, justification = suggest_resolution(row, all_txns)
        resolutions.append(action)
        justifications.append(justification)
        
    result_df = pd.DataFrame({
        "dispute_id": merged_df["dispute_id"],
        "suggested_action": resolutions,
        "justification": justifications
    })
    
    return result_df
