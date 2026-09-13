# Bank Customer Churn — Risk Intelligence Dashboard

Streamlit app for the Unified Mentor Project 1 — Finance Analytics project.

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Launch the app:
   ```
   streamlit run app.py
   ```

3. It will open in your browser at http://localhost:8501

## Folder structure

```
churn_app/
├── app.py                  <- main Streamlit app
├── requirements.txt
├── models/
│   ├── churn_model_final.pkl
│   ├── model_features.pkl
│   └── decision_threshold.pkl
└── data/
    └── European_Bank.csv
```

## Modules included (per project brief)

- Churn risk calculator (with SHAP explanation per prediction)
- Probability distribution visualization
- Feature importance dashboard (native importance + SHAP summary)
- What-if scenario simulator (with sensitivity curve)

Model: LightGBM (tuned), Test ROC-AUC 0.8699, decision threshold 0.321
(tuned from default 0.5 to improve churn recall).
