import pandas as pd
import json
import time
from datetime import timedelta

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from groq import Groq
except ImportError:
    Groq = None

class AgenticPipeline:
    def __init__(self, provider="openai", api_key=None, model=None):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.client = None
        
        if self.provider == "openai" and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.model = self.model or "gpt-3.5-turbo"
        elif self.provider == "groq" and self.api_key:
            self.client = Groq(api_key=self.api_key)
            self.model = self.model or "llama-3.3-70b-versatile"

    def _call_llm(self, system_prompt, user_prompt, is_json=False):
        if not self.client:
            return None
            
        try:
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1
            }
            
            if is_json and self.provider == "openai":
                params["response_format"] = {"type": "json_object"}
                
            completion = self.client.chat.completions.create(**params)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"LLM Call Error: {e}")
            return None

    def analyze_case(self, dispute_row, related_txns_df):
        """
        Analyzes a single dispute context using the LLM.
        """
        # Prepare Context String
        txn_context = "No related transactions found."
        if not related_txns_df.empty:
            txn_context = related_txns_df.to_json(orient="records")
            
        dispute_desc = (
            f"ID: {dispute_row['dispute_id']}, "
            f"Customer: {dispute_row['customer_id']}, "
            f"Amount: {dispute_row['amount']}, "
            f"Description: '{dispute_row['description']}', "
            f"Created At: {dispute_row['created_at']}"
        )
        
        system_prompt = (
            "You are an expert Dispute Resolution Agent. "
            "Your goal is to analyze a customer dispute and their transaction history to decide the outcome.\n"
            "Return a strictly valid JSON object with keys: 'predicted_category', 'suggested_action', 'confidence' (0.0-1.0), and 'explanation'.\n"
            "Categories: [DUPLICATE_CHARGE, FRAUD, FAILED_TRANSACTION, REFUND_PENDING, OTHERS]\n"
            "Actions: [Auto-refund, Manual review, Escalate to bank, Mark as potential fraud, Ask for more info]"
        )
        
        user_prompt = (
            f"Dispute Details:\n{dispute_desc}\n\n"
            f"Transaction History (same user/related):\n{txn_context}\n\n"
            "Analyze the evidence. If the user claims duplicate and you see two successful txns, suggest Auto-refund. "
            "If fraud is claimed, mark as potential fraud. "
            "Provide your JSON decision:"
        )
        
        response = self._call_llm(system_prompt, user_prompt, is_json=True)
        
        # Parse JSON
        try:
            # Clean response for weak LLMs that might add markdown
            if response:
                clean_response = response.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_response)
                return data
            else:
                return {"predicted_category": "ERROR", "suggested_action": "Manual review", "confidence": 0.0, "explanation": "LLM call failed."}
        except json.JSONDecodeError:
            return {"predicted_category": "OTHERS", "suggested_action": "Manual review", "confidence": 0.0, "explanation": f"LLM output parsing failed. Raw: {response[:100] if response else 'None'}"}

    def run_batch(self, disputes_df, transactions_df, progress_callback=None):
        """
        Run the analysis over the full dataframe.
        """
        results = []
        
        # Pre-convert timestamps
        if "timestamp" in transactions_df.columns:
            transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"])
            
        for i, (index, row) in enumerate(disputes_df.iterrows()):
            # Find relevant transactions (Same user, +/- 24 hours or just same user for context)
            # For simplicity in this contextual window, let's pass all txns for that customer
            cust_txns = transactions_df[transactions_df["customer_id"] == row["customer_id"]]
            
            # Analyze
            decision = self.analyze_case(row, cust_txns)
            decision["dispute_id"] = row["dispute_id"]
            results.append(decision)
            
            if progress_callback:
                progress_callback(i + 1, len(disputes_df))
                
        return pd.DataFrame(results)

    def chat_with_data(self, query, processed_df):
        """
        Chat function that uses the *same* LLM to answer questions about the processed data.
        """
        # We can dump a summary of the data into the context if it's small enough, 
        # or use the RAG approach. Since the user wants 'chat option with dataset' 
        # and we are in the 'LLM-only' solution, let's feed the simplified dataset stats 
        # and top rows to the LLM context.
        
        stats = processed_df["predicted_category"].value_counts().to_dict()
        # Fix: Only send the first 10 rows to avoid token limit issues
        sample_data = processed_df.head(10).to_csv(index=False)
        
        system_prompt = (
            "You are a Data Analyst Assistant. You have access to the processed dispute dataset. "
            "Answer the user's questions based on the summary statistics and sample data provided below."
        )
        
        user_prompt = (
            f"Dataset Statistics:\n{stats}\n\n"
            f"Sample Data (First 10 rows):\n{sample_data}\n\n"
            f"User Query: {query}"
        )
        
        return self._call_llm(system_prompt, user_prompt, is_json=False)
