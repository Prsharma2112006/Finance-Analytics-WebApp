# 🏦 Bank Customer Churn — Risk Intelligence Dashboard
**Unified Mentor Virtual Internship | Project 1: Finance Analytics**

> An enterprise-grade predictive machine learning and interactive web application designed to forecast retail banking customer churn, provide quantitative risk scores, and ensure full regulatory explainability.

---

## 📋 Executive Summary
Customer churn directly impacts Customer Lifetime Value (CLV), revenue stability, and long-term competitiveness in retail banking. Traditional analytics only explain *why* customers left after the fact. This project introduces a **Predictive Churn Intelligence System** that assigns quantitative risk probabilities ($0-1$) to customers prior to churn, empowering banking teams to shift from reactive damage control to proactive, targeted retention campaigns.

---

## 🛠️ Technology Stack & Libraries

This project is built using a robust, industry-standard Python data science and deployment stack:

* **Core Language:** Python 3.10+
* **Data Manipulation & Processing:** Pandas, NumPy
* **Machine Learning & Modeling:** Scikit-Learn, LightGBM
* **Model Explainability:** SHAP (SHapley Additive exPlanations)
* **Model Persistence:** Joblib
* **Data Visualization:** Matplotlib, Seaborn
* **Web Application Framework:** Streamlit
* **Version Control:** Git & GitHub

---

## 📊 Key Features & Dashboard Modules

The interactive Streamlit application (`app.py`) features five core analytical modules:

1. **📊 Overview KPI Dashboard:** High-level metrics tracking total customer volume ($10,000$), historical churn rate ($20.4\%$), model ROC-AUC performance ($0.8699$), and high-risk customer counts. Includes an automated table highlighting top highest-risk accounts.
2. **🧮 Churn Risk Calculator:** Real-time customer profiling tool allowing users to adjust credit scores, account balances, age, and product counts to instantly evaluate individual churn likelihood backed by automated SHAP local explainability.
3. **📈 Probability Distribution:** Population-level risk distribution histograms, geographic segment comparisons (France, Spain, Germany), product density breakdowns, and risk-tier segmentations.
4. **🔍 Feature Importance & SHAP Summary:** Global feature rankings via tree split counts and advanced SHAP beeswarm/summary plots detailing the direction and magnitude of feature impacts.
5. **🎛️ What-If Scenario Simulator:** Interactive sensitivity simulator allowing stakeholders to test how altering customer engagement metrics or financial profiles shifts churn probabilities across decision thresholds.

---

## 📈 Model Performance & Methodology

* **Algorithm:** Tuned LightGBM Classifier (Gradient Boosting Framework)
* **Validation Strategy:** Stratified Train-Test Split (preserving $20.37\%$ class balance)
* **Primary Evaluation Metric:** Test **ROC-AUC = 0.8699**
* **Business Decision Threshold:** Optimized at **$0.321$** (rather than a default $0.5$) to properly balance false positives and recall actual churners.
* **Feature Engineering:** Constructed domain-specific indicators including balance-per-product, geography encoding (one-hot), and engagement-product interaction terms.

---

## 📂 Project Directory Structure

```text
Project 1 Finance Analytics/
├── data/
│   ├── raw/
│   │   └── European_Bank.csv
│   └── processed/
├── notebooks/
│   └── 01_data_understanding_eda.ipynb
├── src/
├── models/
│   ├── churn_model_final.pkl
│   ├── decision_threshold.pkl
│   └── model_features.pkl
├── reports/
│   ├── figures/
│   └── documents/
├── app.py
├── requirements.txt
└── .gitignore