import os
import pandas as pd
import getpass
from src.data_loader import load_data, merge_data
from src.classifier import process_classification as rule_classifier
from src.resolver import process_resolutions
from src.ml_classifier import ml_classifier
from src.llm_engine import rag_engine
from src.agentic_flow import AgenticPipeline

def get_api_config(mode_name):
    print(f"\n--- {mode_name} Configuration ---")
    provider = input("Select LLM Provider (openai/groq) [default: openai]: ").strip().lower() or "openai"
    
    while True:
        api_key = getpass.getpass(f"Enter {provider.upper()} API Key: ").strip()
        if not api_key:
            print("Error: API Key cannot be empty. Please try again.")
            continue
        
        # Validation Check
        print(f"Verifying {provider.upper()} API key...")
        verification_agent = AgenticPipeline(provider=provider, api_key=api_key)
        # Try a tiny response to check if key works
        test_resp = verification_agent._call_llm("Respond with 'OK'", "Test", is_json=False)
        
        if test_resp:
            print("✅ API Key verified successfully.")
            return provider, api_key
        else:
            print("❌ API Key verification failed. Please check your key and provider.")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return provider, None

def run_standard_mode(disputes, transactions):
    print("\nRunning Standard (Rule-Based) Mode...")
    classified_df = rule_classifier(disputes)
    
    classified_full = disputes.merge(classified_df, on="dispute_id")
    merged_data = merge_data(classified_full, transactions)
    resolutions_df = process_resolutions(merged_data, transactions)
    
    final_df = merged_data.merge(resolutions_df, on="dispute_id")
    
    output_path = "standard_resolutions.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Success! Results saved to {output_path}")
    return final_df

def run_advanced_mode(disputes, transactions):
    print("\nRunning Advanced (ML & AI) Mode...")
    
    # 1. ML Classification
    if not ml_classifier.is_trained:
        print("Training ML Model...")
        ml_classifier.train()
        
    print("Classifying disputes with ML...")
    descriptions = disputes["description"].fillna("").tolist()
    results = ml_classifier.predict(descriptions)
    categories, confidences = zip(*results)
    
    classified_df = disputes.copy()
    classified_df["predicted_category"] = categories
    classified_df["confidence"] = confidences
    classified_df["explanation"] = "Classified by SVM Model based on text patterns."
    
    # 2. Resolution logic
    merged_data = merge_data(classified_df, transactions)
    resolutions_df = process_resolutions(merged_data, transactions)
    final_df = merged_data.merge(resolutions_df, on="dispute_id")
    
    output_path = "advanced_resolutions.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Success! Results saved to {output_path}")
    
    # 3. Chat Option
    chat_option = input("\nWould you like to enter AI Chat mode? (y/n): ").strip().lower()
    if chat_option == 'y':
        provider, api_key = get_api_config("Advanced Chat")
        if not api_key:
            print("Skipping chat due to missing API key.")
            return final_df
            
        print("\n--- AI RAG Chat (Type 'exit' to quit) ---")
        rag_engine.ingest_data() # Ensure data is loaded into RAG
        while True:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                break
            print("Assistant is thinking...", end="\r")
            response = rag_engine.query_ai(query, provider=provider, api_key=api_key)
            print(" " * 30, end="\r") # Clear the thinking line
            print(f"AI: {response}")
            
    return final_df

def run_agentic_mode(disputes, transactions):
    print("\nRunning Agentic (LLM-Only) Mode...")
    provider, api_key = get_api_config("Agentic Pipeline")
    
    if not api_key:
        print("Error: API Key is required for Agentic mode.")
        return None

    agent = AgenticPipeline(provider=provider, api_key=api_key)
    
    print("Processing disputes via LLM Agent (this may take a while)...")
    results_df = agent.run_batch(disputes, transactions, progress_callback=lambda c, t: print(f"Progress: {c}/{t}", end="\r"))
    print("\nProcessing complete.")
    
    full_view = disputes.merge(results_df, on="dispute_id")
    
    output_path = "agentic_resolutions.csv"
    full_view.to_csv(output_path, index=False)
    print(f"Success! Results saved to {output_path}")
    
    # Chat Option
    chat_option = input("\nWould you like to chat with the Agent about these results? (y/n): ").strip().lower()
    if chat_option == 'y':
        print("\n--- Agentic Chat (Type 'exit' to quit) ---")
        while True:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                break
            
            print("Agent is analyzing...", end="\r")
            response = agent.chat_with_data(query, full_view)
            print(" " * 30, end="\r") # Clear
            
            if response is None:
                print("Agent: I'm sorry, I encountered an error communicating with the AI service.")
            else:
                print(f"Agent: {response}")
            
    return full_view

def main():
    print("="*40)
    print("   FinAI Dispute Assistant - CLI")
    print("="*40)
    
    print("Loading data...")
    disputes, transactions = load_data()
    
    if disputes is None or transactions is None:
        print("Data load failed. Please ensure data/disputes.csv and data/transactions.csv exist.")
        return

    while True:
        print("\nSelect System Mode:")
        print("1. Standard (Rule-Based)")
        print("2. Advanced (ML & AI)")
        print("3. Agentic (LLM-Only)")
        print("q. Quit")
        
        choice = input("\nEnter choice (1-3 or q): ").strip().lower()
        
        if choice == '1':
            run_standard_mode(disputes, transactions)
        elif choice == '2':
            run_advanced_mode(disputes, transactions)
        elif choice == '3':
            run_agentic_mode(disputes, transactions)
        elif choice == '4' or choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
