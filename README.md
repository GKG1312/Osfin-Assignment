# AI-Powered Dispute Assistant

This project is a comprehensive AI assistant designed to help financial support teams resolve payment disputes. It features three distinct architectural approaches to solving the problem, ranging from deterministic rules to fully generative AI agents.

## 🚀 Features & Solutions

The application offers three modes, switchable via the sidebar:

### 1. Standard (Rule-Based)
*   **Best for**: Speed, Determinism, Low Cost.
*   **Mechanism**: Uses keyword matching (Regex) to classify disputes and hard-coded logic trees to suggest resolutions.
*   **Chat**: Simple keyword-based bot.

### 2. Advanced (Hybrid ML & AI)
*   **Best for**: Scalability, Accuracy, Context Awareness.
*   **Mechanism**:
    *   **Classification**: Scikit-Learn Pipeline (TF-IDF + SGD Classifier) trained on the dataset.
    *   **Resolution**: Human-defined logic combined with ML outputs.
    *   **Chat**: **RAG (Retrieval-Augmented Generation)** engine that indexes disputes for semantic search.
*   **Smart Features**: Includes "Fuzzy Duplicate Detection" using RAG heuristics to find claims like "charged twice" vs "two debits".

### 3. Agentic (LLM-Only)
*   **Best for**: Complex reasoning, Unstructured data, Zero-shot capability.
*   **Mechanism**: A single **LLM Agent** (Powered by **Groq/Llama-3** or **OpenAI/GPT-4**) that reads the dispute + transaction history and decides everything in one pass.
*   **Chat**: Full conversational capabilities with the dataset.

---

## 🛠️ Project Structure

```
osfin/
├── data/                   # Input CSV files
├── src/                    # Core Logic
│   ├── classifier.py       # Rule-based logic
│   ├── ml_classifier.py    # Scikit-Learn Pipeline
│   ├── agentic_flow.py     # LLM Agent Pipeline
│   ├── llm_engine.py       # RAG Engine
│   ├── resolver.py         # Resolution Helper
│   └── helper...
├── app.py                  # Streamlit Main Application
└── requirements.txt        # Dependencies
```

## ⚡ Setup & Running

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Select Mode**:
   Use the Sidebar to toggle between Rule-Based, Advanced (ML), and Agentic modes.
   For AI features, you can provide an **OpenAI** or **Groq** API Key.

## 📊 Key Implementation Details
- **Duplicate Detection**: In Standard/Advanced modes, we use heuristics to check specifically for `SUCCESS` transactions from the same user within a 1-hour window.
- **RAG Chat**: The Advanced mode uses specific heuristics to capture "list all" queries, bypassing standard vector limits to ensure accurate reporting (e.g., "List all duplicate charges").
