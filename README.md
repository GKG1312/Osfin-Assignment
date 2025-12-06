# AI-Powered Dispute Assistant

This project is an AI assistant designed to help financial support teams resolve payment disputes. It automates the classification of customer complaints and suggests appropriate resolutions based on transaction data.

## Features

- **Dispute Classification**: Automatically categorizes disputes into `DUPLICATE_CHARGE`, `FRAUD`, `FAILED_TRANSACTION`, `REFUND_PENDING`, etc.
- **Resolution Engine**: Analyzes transaction status and history to suggest actions (e.g., "Auto-refund", "Escalate").
- **Interactive Dashboard**: A Streamlit-based UI for agents to explore disputes and view analytics.
- **Natural Language Querying**: Simulate asking questions about the dispute data.

## Project Structure

```
osfin/
├── data/                   # Input CSV files
│   ├── disputes.csv
│   └── transactions.csv
├── src/                    # Source code
│   ├── analytics.py        # Analytics functions
│   ├── classifier.py       # Rule-based classification logic
│   ├── data_loader.py      # Data loading and merging
│   └── resolver.py         # Resolution suggestion logic
├── app.py                  # Streamlit Dashboard application
├── main_cli.py             # CLI script to generate CSV outputs
├── classified_disputes.csv # Generated output from Task 1
├── resolutions.csv         # Generated output from Task 2
└── README.md               # This file
```

## Setup & Running

1. **Install Dependencies**:
   ```bash
   pip install pandas streamlit plotly
   ```

2. **Generate Reports (CLI)**:
   Run the main script to process the data and generate `classified_disputes.csv` and `resolutions.csv`.
   ```bash
   python main_cli.py
   ```

3. **Launch Dashboard**:
   Start the interactive web interface.
   ```bash
   streamlit run app.py
   ```

## methodology

- **Classification**: Uses keyword analysis on the dispute description to identify intent.
- **Resolution**: Combines the predicted category with the actual transaction status (from `transactions.csv`) and history (duplicate detection) to recommend the next best action.
