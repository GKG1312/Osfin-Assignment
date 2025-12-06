import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_data, merge_data
from src.classifier import process_classification
from src.resolver import process_resolutions
from src.analytics import get_disputes_by_type, get_unresolved_fraud

# Page Config
st.set_page_config(page_title="AI Dispute Assistant", layout="wide")

# Title
st.title("🤖 AI-Powered Dispute Resolution Assistant")

# Load Data
@st.cache_data
def get_processed_data():
    disputes, transactions = load_data()
    if disputes is None:
        return None, None, None
        
    # Run pipeline (simulation)
    classified = process_classification(disputes)
    classified_full = disputes.merge(classified, on="dispute_id")
    merged = merge_data(classified_full, transactions)
    resolutions = process_resolutions(merged, transactions)
    
    # Final combined view
    final_df = merged.merge(resolutions, on="dispute_id")
    return final_df, disputes, transactions

df, raw_disputes, raw_txns = get_processed_data()

if df is None:
    st.error("Failed to load data. Please check data/ directory.")
    st.stop()

# Sidebar for Navigation
page = st.sidebar.radio("Navigate", ["Dashboard", "Dispute Explorer", "Ask AI (Simulated)"])

if page == "Dashboard":
    st.header("📊 Overview")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Disputes", len(df))
    col2.metric("Fraud Cases", len(df[df["predicted_category"] == "FRAUD"]))
    col3.metric("Auto-Refunds", len(df[df["suggested_action"] == "Auto-refund"]))
    col4.metric("Escalations", len(df[df["suggested_action"] == "Escalate to bank"]))
    
    # Charts
    st.subheader("Disputes by Category")
    counts = df["predicted_category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    fig = px.bar(counts, x="Category", y="Count", color="Category", title="Dispute Volume by Type")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Dispute Trends")
    # Parse dates
    df["created_at"] = pd.to_datetime(df["created_at"])
    daily = df.groupby(df["created_at"].dt.date).size().reset_index(name="Count")
    fig2 = px.line(daily, x="created_at", y="Count", title="Daily Dispute Intake")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Dispute Explorer":
    st.header("🔍 Dispute Resolution center")
    
    # Filters
    cat_filter = st.multiselect("Filter by Category", df["predicted_category"].unique())
    if cat_filter:
        view_df = df[df["predicted_category"].isin(cat_filter)]
    else:
        view_df = df
        
    st.dataframe(view_df[[
        "dispute_id", "customer_id_dispute", "amount_dispute", 
        "predicted_category", "confidence", "suggested_action", "justification"
    ]], use_container_width=True)
    
    # Detail View
    st.subheader("Detailed Analysis")
    selected_id = st.selectbox("Select Dispute ID", view_df["dispute_id"].unique())
    
    if selected_id:
        row = view_df[view_df["dispute_id"] == selected_id].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Description:** {row['description']}")
            st.write(f"**Transaction Status:** {row.get('status', 'Unknown')}")
            st.write(f"**Transaction Time:** {row.get('timestamp', 'Unknown')}")
        with c2:
            st.success(f"**AI Recommendation:** {row['suggested_action']}")
            st.write(f"**Reasoning:** {row['justification']}")
            st.warning(f"**Confidence:** {row['confidence']}")

elif page == "Ask AI (Simulated)":
    st.header("💬 AI Command Interface")
    st.write("Try asking: 'How many duplicate charges today?', 'List unresolved fraud disputes', or 'Break down disputes by type'")
    
    query = st.text_input("Enter your query:")
    
    if query:
        query_lower = query.lower()
        
        if "duplicate" in query_lower and "count" in query_lower or "how many" in query_lower:
            count = len(df[df["predicted_category"] == "DUPLICATE_CHARGE"])
            st.markdown(f"### 🤖 Answer: There are **{count}** duplicate charge disputes.")
            
        elif "fraud" in query_lower and "list" in query_lower:
            fraud = df[df["predicted_category"] == "FRAUD"][["dispute_id", "description", "amount_dispute"]]
            st.markdown("### 🤖 Here are the fraud cases:")
            st.table(fraud)
            
        elif "break down" in query_lower or "by type" in query_lower:
            st.markdown("### 🤖 Dispute Breakdown:")
            st.write(df["predicted_category"].value_counts())
            
        else:
            st.markdown("### 🤖 I didn't verify that query pattern. Try the examples above!")

