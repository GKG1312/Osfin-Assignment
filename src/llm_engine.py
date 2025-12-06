import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from groq import Groq
except ImportError:
    Groq = None

class DisputeSimpleRAG:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.context_data = [] # List of strings
        self.doc_ids = []
        self.tfidf_matrix = None
        
    def ingest_data(self):
        """
        Loads CSVs and creates a text 'knowledge base'.
        """
        try:
            disputes = pd.read_csv(os.path.join(self.data_dir, "disputes.csv"))
            transactions = pd.read_csv(os.path.join(self.data_dir, "transactions.csv"))
            
            # Merge to create rich context
            # Left join to preserve all disputes
            merged = disputes.merge(transactions, on="txn_id", how="left", suffixes=("", "_txn"))
            
            self.context_data = []
            self.doc_ids = []
            
            for _, row in merged.iterrows():
                # Create a narrative for each case
                doc = (
                    f"Dispute ID: {row['dispute_id']}. "
                    f"Customer {row['customer_id']} reported: '{row['description']}'. "
                    f"Transaction ID: {row['txn_id']}. "
                    f"Amount: {row['amount']}. "
                    f"Status: {row.get('status', 'Unknown')}. "
                    f"Category: {row.get('predicted_category', 'Unclassified')}. "
                    f"Timestamp: {row.get('timestamp', 'Unknown')}."
                )
                self.context_data.append(doc)
                self.doc_ids.append(row['dispute_id'])
                
            # Fit vectorizer
            if self.context_data:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.context_data)
                
        except Exception as e:
            print(f"RAG Ingestion Failed: {e}")

    def retrieve(self, query, top_k=5):
        """
        Retrieves relevant context.
        Smartly detects if user is asking for a 'list' or 'count' of a specific category
        and retrieves ALL matching records if so, bypassing top_k limits.
        """
        if self.tfidf_matrix is None:
            self.ingest_data()
            
        if not self.context_data:
            return []

        query_lower = query.lower()
        
        # Smart Heuristic: If user asks for "list all duplicate" or "show duplicates",
        # we manually filter the source data because vector search might only give top 3-5.
        if "duplicate" in query_lower and ("list" in query_lower or "all" in query_lower or "show" in query_lower):
            # Manually find all indices where text indicates duplicate
            # We can check specific keywords in the doc string
            results = []
            for doc in self.context_data:
                # Check for category explicitly if we added it to doc, or keywords
                if "Category: DUPLICATE_CHARGE" in doc or \
                   any(k in doc.lower() for k in ["charged twice", "double payment", "duplicate", "two debits", "two debit", "two upi debit"]):
                    results.append(doc)
            # Return up to 20 to fit in context window
            return results[:20]

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if sims[idx] > 0.1: # Threshold to ignore total noise
                results.append(self.context_data[idx])
        return results

    def query_ai(self, user_query, provider=None, api_key=None):
        """
        Orchestrates retrieval and generation.
        provider: 'openai' or 'groq' or None
        """
        # 1. Retrieve Context
        context_docs = self.retrieve(user_query)
        context_str = "\n\n".join(context_docs)
        
        system_prompt = (
            "You are an expert Dispute Resolution AI for a Fintech Company. "
            "Use the provided CONTEXT data to answer the user's question accurately. "
            "If the answer is found in the context, cite the Dispute ID. "
            "If not found, use general financial knowledge but explicitly state you don't see it in the internal records."
        )
        
        prompt_content = f"Context Data:\n{context_str}\n\nUser Question: {user_query}"

        # 2. Call LLM
        if provider == "openai" and api_key and OpenAI:
            try:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_content}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"OpenAI Error: {str(e)}"

        elif provider == "groq" and api_key and Groq:
            try:
                client = Groq(api_key=api_key)
                # Llama 3 prompt formatting is handled by the API usually, but we stick to standard chat structure
                # llama-3.3-70b-versatile is the requested model
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                return completion.choices[0].message.content
            except Exception as e:
                return f"Groq Error: {str(e)}"

        # 3. Fallback / Simulation (Context Aware but no Generative AI)
        if not context_docs:
            return "I couldn't find any specific records in the database matching your query based on keywords."
        
        return (
            f"**Data Retrieved via RAG ({len(context_docs)} records):**\n\n"
            f"{context_str}\n\n"
            f"*(System Note: No LLM connected. Showing raw matched records. Connect OpenAI or Groq for natural language answers.)*"
        )

rag_engine = DisputeSimpleRAG()
