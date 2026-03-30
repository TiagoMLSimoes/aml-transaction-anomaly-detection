# AML Transaction Anomaly Detection

Unsupervised anomaly detection system for financial transactions using Isolation Forest — built to identify suspicious patterns consistent with money laundering and fraud.

---

## What This Project Does

Financial institutions process millions of transactions daily. Manual review is impossible at scale. This system automatically profiles each customer's behavioral patterns and flags deviations that may indicate:

- **Smurfing** — high-frequency small transactions to avoid reporting thresholds
- **Layering** — transfers to multiple unique destinations
- **Cash-out structuring** — disproportionate cash withdrawal ratios
- **Balance manipulation** — discrepancies between expected and actual balance deltas

---

## Technical Stack

| Component | Technology | Reason |
|---|---|---|
| Data storage | SQLite | Portable, production-simulatable, SQL-queryable |
| Feature engineering | SQL (aggregations) | Behavioral profiling per customer |
| ML Model | Isolation Forest (Scikit-learn) | Unsupervised — no fraud labels required |
| Preprocessing | StandardScaler | Prevents feature dominance by scale |
| Visualization | Matplotlib + Seaborn | Risk distribution, top alerts, correlation heatmap |

---

## Why Isolation Forest?

Most fraud datasets are **unlabeled** — you don't know which transactions are fraud until after investigation. Isolation Forest is non-supervised: it learns what "normal" looks like and flags statistical outliers without requiring labeled examples.

**Core intuition:** Anomalous data points are isolated in fewer random partition steps than normal points. The algorithm builds 200 random trees and measures how quickly each data point gets isolated. Fast isolation = high anomaly score.

---

## Features Engineered (per customer)

All features computed via SQL aggregations over transaction history:

- `total_transactions` — transaction frequency
- `avg_amount`, `max_amount`, `total_amount` — value profile
- `std_amount` — value variability
- `avg_balance_before`, `avg_balance_delta` — balance behavior
- `cashout_count`, `cashout_ratio` — cash withdrawal pattern
- `transfer_count` — layering indicator
- `unique_destinations` — destination diversity (structuring indicator)

---

## Results

The model processes 500,000 transactions and produces:

- `outputs/aml_alerts.csv` — ranked list of flagged customers with risk level (HIGH/MEDIUM/LOW) and continuous risk score
- `outputs/full_results.csv` — complete dataset with anomaly flags
- `outputs/aml_analysis_charts.png` — risk score distributions, top 20 alerts, frequency vs volume scatter, cash-out ratio comparison
- `outputs/feature_correlation_heatmap.png` — feature correlation matrix

---

## Setup

```bash
# 1. Install dependencies
pip install pandas scikit-learn matplotlib seaborn sqlalchemy

# 2. Download dataset
# https://www.kaggle.com/datasets/ealaxi/paysim1
# Rename to paysim.csv and place in project root

# 3. Run
python project1_aml_detection.py
```

---

## Dataset

**PaySim** — synthetic mobile money transaction dataset based on real transaction logs from a mobile money service in Africa. Contains ~6.3M transactions across 30 days.

Source: [Kaggle — PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

## Key Design Decisions

**Why SQLite instead of in-memory Pandas for feature engineering?**  
SQL aggregations mirror production environments where transaction data lives in relational databases. Demonstrating SQL proficiency alongside Python is intentional.

**Why contamination=0.01?**  
Real-world fraud rates in financial datasets range from 0.1% to 2%. 1% is a conservative midpoint. In production, this threshold would be calibrated against investigation capacity and regulatory requirements.

**Why StandardScaler before Isolation Forest?**  
Although Isolation Forest is theoretically scale-invariant, features with extreme value ranges (0 to 10M for amounts vs 1 to 50 for counts) can bias the random split selection. Scaling ensures uniform feature representation.

---

## Extensions (Production Roadmap)

- [ ] Replace SQLite with PostgreSQL or Azure SQL
- [ ] Add SHAP values for explainability per alert
- [ ] Deploy as REST API (FastAPI)
- [ ] Containerize with Docker
- [ ] Add automated retraining pipeline (Azure ML)
- [ ] Integrate with Power BI for compliance dashboard

---

*Part of a financial data engineering portfolio. See also: [Project 2 — Automated Reporting Pipeline] | [Project 3 — Customer Risk Scoring API]*
