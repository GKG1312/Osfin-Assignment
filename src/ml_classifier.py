import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

class DisputeClassifierML:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
            ('clf', SGDClassifier(loss='modified_huber', random_state=42)) # modified_huber gives probability estimates
        ])
        self.is_trained = False

    def train(self, data_path="classified_disputes.csv", fallback_data=None):
        """
        Trains the model.
        If structured data exists, use it.
        Otherwise use a small fallback dataset for cold start.
        """
        X = []
        y = []

        # Try to load existing classified data (bootstrapping from the rule-based output)
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            # Ensure we have the right columns
            if "description" in df.columns: # Merge logic might be needed if description isn't in classified csv
                pass # Depending on how it was saved.
            # actually classified_disputes.csv from previous step only had id, category, etc. 
            # We need to merge with disputes.csv to get description.
            pass
        
        # Fallback / Synthetic Data for robust "Demo" Training
        # This ensures the ML model behaves somewhat intelligently even with 20 rows
        synthetic_data = [
            ("I was charged twice for the same thing", "DUPLICATE_CHARGE"),
            ("Duplicate debit occurred", "DUPLICATE_CHARGE"),
            ("Two transactions for one purchase", "DUPLICATE_CHARGE"),
            ("charged double", "DUPLICATE_CHARGE"),
            
            ("This is a fraud transaction", "FRAUD"),
            ("I did not authorize this payment", "FRAUD"),
            ("Suspicious activity on my card", "FRAUD"),
            ("stolen card usage", "FRAUD"),
            
            ("The payment failed but money was cut", "FAILED_TRANSACTION"),
            ("Transaction status failed but debited", "FAILED_TRANSACTION"),
            ("money not received by merchant", "FAILED_TRANSACTION"),
            ("stuck payment", "FAILED_TRANSACTION"),
            
            ("waiting for my refund", "REFUND_PENDING"),
            ("refund not processed yet", "REFUND_PENDING"),
            ("cancelled order still charged", "REFUND_PENDING"),
            ("return request refund pending", "REFUND_PENDING"),
            
            ("I really don't know what this is", "OTHERS"),
            ("random check", "OTHERS")
        ]
        
        # Load real data if available to augment
        # We assume 'disputes.csv' and 'classified_disputes.csv' are available
        try:
            disputes = pd.read_csv("data/disputes.csv")
            if os.path.exists("classified_disputes.csv"):
                classified = pd.read_csv("classified_disputes.csv")
                merged = disputes.merge(classified, on="dispute_id")
                
                real_X = merged["description"].tolist()
                real_y = merged["predicted_category"].tolist()
                
                X.extend(real_X)
                y.extend(real_y)
        except Exception as e:
            print(f"Warning: Could not load local csv for training: {e}")
            
        # Add synthetic data
        syn_X, syn_y = zip(*synthetic_data)
        X.extend(syn_X)
        y.extend(syn_y)
        
        # Train
        self.model.fit(X, y)
        self.is_trained = True
        return self.model.score(X, y)

    def predict(self, descriptions):
        if not self.is_trained:
            self.train()
            
        preds = self.model.predict(descriptions)
        probs = self.model.predict_proba(descriptions)
        
        results = []
        for pred, prob in zip(preds, probs):
            # Get max probability
            confidence = np.max(prob)
            results.append((pred, confidence))
            
        return results

# Singleton for easy import
ml_classifier = DisputeClassifierML()
