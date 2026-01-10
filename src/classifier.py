import pandas as pd
import re

def classify_dispute(description):
    """
    Classifies a single dispute description into categories.
    Returns: category, confidence, explanation
    """
    desc_lower = str(description).lower()
    
    # Priority 1: Fraud (High severity)
    if any(keyword in desc_lower for keyword in ["fraud", "suspicious", "unauthorized", "didn't make", "recognize", "not authorize"]):
        return "FRAUD", 0.95, "User explicitly mentioned fraud or unauthorized transaction."
    
    # Priority 2: Duplicate
    if any(keyword in desc_lower for keyword in ["twice", "double", "duplicate", "two debit", "two upi", "two upi debit"]):
        return "DUPLICATE_CHARGE", 0.90, "User mentions multiple charges or duplication."
    
    # Priority 3: Refund Pending
    if any(keyword in desc_lower for keyword in ["refund", "waiting", "canceled", "cancelled", "return"]):
        return "REFUND_PENDING", 0.85, "User is waiting for a refund or mentioned cancellation."
    
    # Priority 4: Failed Transaction
    if any(keyword in desc_lower for keyword in ["failed", "stuck", "debited", "not received", "fail", "wrong beneficiary"]):
        return "FAILED_TRANSACTION", 0.85, "User mentions transaction failure or money debited without success."
        
    return "OTHERS", 0.50, "No specific keywords matched."

def process_classification(disputes_df):
    """
    Applies classification to the disputes dataframe.
    """
    results = disputes_df["description"].apply(classify_dispute)
    
    # Expand the results into separate columns
    # We maintain the original index to assign back correctly
    classified_df = disputes_df.copy()
    
    # ZIP the results to unpack
    categories, confidences, explanations = zip(*results)
    
    classified_df["predicted_category"] = categories
    classified_df["confidence"] = confidences
    classified_df["explanation"] = explanations
    
    return classified_df[["dispute_id", "predicted_category", "confidence", "explanation"]]
