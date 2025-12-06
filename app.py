import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Import Logic modules
from src.data_loader import load_data, merge_data
from src.classifier import process_classification as rule_classifier
from src.resolver import process_resolutions
from src.analytics import get_disputes_by_type
from src.ml_classifier import ml_classifier
from src.llm_engine import rag_engine

# Page Config
st.set_page_config(page_title="FinAI Dispute Assistant", layout="wide", page_icon="🤖")
# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("Configuration")
mode = st.sidebar.radio("System Mode", ["Standard (Rule-Based)", "Advanced (ML & AI)"])

api_key = ""
provider = "openai"

if mode == "Advanced (ML & AI)":
    st.sidebar.markdown("### 🧠 AI Settings")
    provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Groq"])
    
    if provider == "OpenAI":
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    else:
        api_key = st.sidebar.text_input("Groq API Key", type="password")
        
    st.sidebar.info(f"Model: {provider} based RAG Assistant")

# --- DATA LOADING ---
@st.cache_data
def get_data(use_ml=False):
    disputes, transactions = load_data()
    if disputes is None:
        return None, None, None
    
    # Classification Step
    if use_ml:
        # ML Pipeline
        # Ensure model is trained
        if not ml_classifier.is_trained:
            ml_classifier.train()
            
        descriptions = disputes["description"].fillna("").tolist()
        results = ml_classifier.predict(descriptions)
        
        # Unpack results
        categories, confidences = zip(*results)
        
        classified_df = disputes.copy()
        classified_df["predicted_category"] = categories
        classified_df["confidence"] = confidences
        # ML doesn't give text explanation by default, simplified
        classified_df["explanation"] = "Classified by SVM Model based on text patterns."
        
        # Select just the cols we need to merge
        classified_subset = classified_df[["dispute_id", "predicted_category", "confidence", "explanation"]]
    else:
        # Rule Based
        classified_subset = rule_classifier(disputes)
    
    # Merge & Resolve
    classified_full = disputes.merge(classified_subset, on="dispute_id")
    merged = merge_data(classified_full, transactions)
    resolutions = process_resolutions(merged, transactions)
    
    final_df = merged.merge(resolutions, on="dispute_id")
    return final_df, disputes, transactions

# Run Pipeline based on Mode
use_ml_mode = (mode == "Advanced (ML & AI)")
df, raw_disputes, raw_txns = get_data(use_ml=use_ml_mode)

if df is None:
    st.error("Data load failed.")
    st.stop()

# --- MAIN APP UI ---
st.title("🤖 FinAI Dispute Resolution")
st.markdown(f"**Current Mode:** `{mode}`")

# Navigation
tabs = st.tabs(["📊 Dashboard", "🔍 Case Explorer", "💬 AI Assistant"])

with tabs[0]:
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cases", len(df))
    fraud_count = len(df[df["predicted_category"] == "FRAUD"])
    c2.metric("Fraud Detected", fraud_count, delta_color="inverse")
    
    auto_ref = len(df[df["suggested_action"] == "Auto-refund"])
    c3.metric("Auto-Refunds", auto_ref)
    
    avg_conf = df["confidence"].mean()
    c4.metric("Avg Model Confidence", f"{avg_conf:.1%}")
    
    # Charts
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        counts = df["predicted_category"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        fig = px.pie(counts, values="Count", names="Category", title="Dispute Types", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with c_chart2:
        # Status distribution
        status_counts = df["suggested_action"].value_counts().reset_index()
        status_counts.columns = ["Action", "Count"]
        fig2 = px.bar(status_counts, x="Action", y="Count", color="Action", title="Suggested Actions")
        st.plotly_chart(fig2, use_container_width=True)

with tabs[1]:
    st.markdown("### Dispute Case Management")
    
    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        cat_filter = st.multiselect("Filter Category", df["predicted_category"].unique())
    with col_f2:
        conf_filter = st.slider("Min Confidence", 0.0, 1.0, 0.5)
        
    view_df = df[df["confidence"] >= conf_filter]
    if cat_filter:
        view_df = view_df[view_df["predicted_category"].isin(cat_filter)]
        
    # Rename cols for display if needed or just use correct ones
    # 'amount' might be 'amount_dispute' after merge
    display_cols = ["dispute_id", "description", "predicted_category", "confidence", "suggested_action"]
    if "amount" in view_df.columns:
        display_cols.insert(2, "amount")
    elif "amount_dispute" in view_df.columns:
        display_cols.insert(2, "amount_dispute")
        
    st.dataframe(
        view_df[display_cols],
        use_container_width=True,
        hide_index=True
    )
    
    if not view_df.empty:
        st.divider()
        st.subheader("Selected Case Details")
        sel_id = st.selectbox("Select Dispute to Inspect", view_df["dispute_id"].unique())
        
        row = view_df[view_df["dispute_id"] == sel_id].iloc[0]
        
        # Safe access helper
        def get_val(r, keys, default="N/A"):
            for k in keys:
                if k in r:
                    return r[k]
            return default
            
        d1, d2 = st.columns(2)
        with d1:
            st.info(f"**Dispute:** {row['description']}")
            amt = get_val(row, ['amount', 'amount_dispute'])
            chan = get_val(row, ['channel', 'channel_dispute'])
            st.write(f"**Amount:** {amt} | **Channel:** {chan}")
            
            status = get_val(row, ['status', 'status_txn', 'status_dispute'])
            st.write(f"**Txn Status:** `{status}`")
        with d2:
            st.success(f"**Recommendation:** {row['suggested_action']}")
            st.caption(f"Reasoning: {row['justification']}")
            st.progress(float(row['confidence']), text=f"Confidence: {row['confidence']:.2f}")

with tabs[2]:
    st.header("💬 AI Assistant")
    
    if use_ml_mode:
        st.markdown("""
        **Context-Aware RAG Assistant**
        Ask questions about specific disputes or general trends. 
        *Examples: "Why was D005 rejected?", "Show me fraud disputes over 5000", "Details of C001"*
        """)
        
        if "messages_ml" not in st.session_state:
            st.session_state.messages_ml = []

        for msg in st.session_state.messages_ml:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask the Advanced AI..."):
            st.session_state.messages_ml.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing Knowledge Base..."):
                    # Use RAG Engine
                    response = rag_engine.query_ai(prompt, provider=provider.lower() if provider else None, api_key=api_key if api_key else None)
                    st.markdown(response)
            
            st.session_state.messages_ml.append({"role": "assistant", "content": response})
            
    else:
        # Rule-Based Chat Implementation
        st.markdown("""
        **Rule-Based Keyword Assistant**
        Try asking: 'How many duplicate charges today?', 'List unresolved fraud disputes', or 'Break down disputes by type'.
        """)
        
        if "messages_rb" not in st.session_state:
            st.session_state.messages_rb = []
            
        for msg in st.session_state.messages_rb:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if query := st.chat_input("Ask Rule-Bot..."):
            st.session_state.messages_rb.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            query_lower = query.lower()
            response = "I didn't verify that query pattern. Try the examples above!"
            
            if "duplicate" in query_lower and ("count" in query_lower or "how many" in query_lower):
                count = len(df[df["predicted_category"] == "DUPLICATE_CHARGE"])
                response = f"There are **{count}** duplicate charge disputes."
                
            elif "fraud" in query_lower and "list" in query_lower:
                fraud = df[df["predicted_category"] == "FRAUD"][["dispute_id", "description"]]
                if not fraud.empty:
                    response = "Here are the fraud cases:\n\n" + fraud.to_markdown()
                else:
                    response = "No fraud cases found."
                
            elif "break down" in query_lower or "by type" in query_lower:
                breakdown = df["predicted_category"].value_counts().to_markdown()
                response = f"Dispute Breakdown:\n\n{breakdown}"
            
            with st.chat_message("assistant"):
                st.markdown(response)
                
            st.session_state.messages_rb.append({"role": "assistant", "content": response})

