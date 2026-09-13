"""
Bank Customer Churn — Risk Intelligence Dashboard
Predictive Modeling and Risk Scoring for Bank Customer Churn

Unified Mentor Project 1 — Finance Analytics

Run with:
streamlit run app.py
"""

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Churn Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# COLOR PALETTE
# ============================================================

BG = "#080F1F"
SIDEBAR_BG = "#070D1B"

PANEL = "#111C30"
PANEL_2 = "#162238"
PANEL_INPUT = "#182438"

GRID = "#293951"
GRID_FAINT = "#3A506D"

TEXT = "#CBD5E1"
TEXT_HEAD = "#F8FAFC"
MUTED = "#94A3B8"

ACCENT = "#14B8A6"       # Primary Teal
ACCENT_2 = "#22D3EE"     # Cyan
GOOD = "#10B981"         # Green
RISK = "#F43F5E"         # Red/Rose
WARN = "#F59E0B"         # Amber
PURPLE = "#A78BFA"       # Analytics Purple


# ============================================================
# MATPLOTLIB GLOBAL SETTINGS
# ============================================================

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID_FAINT,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT_HEAD,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "grid.color": GRID_FAINT,
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# CUSTOM CSS — PREMIUM DASHBOARD DESIGN
# ============================================================

st.markdown(
    f"""
<style>

    /* ========================================================
       GLOBAL APPLICATION & HEADER FIX
       ======================================================== */

    .stApp {{
        background-color: {BG};
    }}

    header, [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}

    .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT_HEAD} !important;
        font-weight: 700 !important;
    }}

    h1 {{
        font-size: 2.4rem !important;
        letter-spacing: -0.04em;
        margin-bottom: 0.6rem !important;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }}

    h2 {{
        font-size: 1.65rem !important;
        margin-top: 1.8rem !important;
    }}

    h3 {{
        font-size: 1.25rem !important;
    }}

    h4 {{
        font-size: 1.05rem !important;
    }}

    p {{
        color: {TEXT};
    }}

    hr {{
        border-color: {GRID};
        margin: 1.5rem 0;
    }}


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG} !important;
        border-right: 1px solid rgba(255,255,255,0.08);
        min-width: 310px !important;
    }}

    [data-testid="stSidebar"] .block-container {{
        padding: 1.8rem 1.3rem;
    }}

    /* Sidebar Brand */

    .sidebar-brand {{
        font-size: 1.28rem;
        font-weight: 750;
        letter-spacing: -0.035em;
        white-space: nowrap;
        margin-bottom: 1.8rem;
    }}

    .sidebar-brand .brand-white {{
        color: {TEXT_HEAD};
    }}

    .sidebar-brand .brand-teal {{
        color: {ACCENT_2};
    }}


    /* Navigation */

    [data-testid="stSidebar"] .stRadio {{
        margin-top: 1rem;
    }}

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{
        gap: 5px;
    }}

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {{
        background-color: transparent !important;
        padding: 0.78rem 0.85rem !important;
        margin-bottom: 0.15rem !important;
        border-radius: 10px !important;
        width: 100% !important;
        color: {MUTED} !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
        border-left: 3px solid transparent !important;
        cursor: pointer;
    }}

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {{
        background-color: rgba(255,255,255,0.045) !important;
        color: {TEXT_HEAD} !important;
    }}

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {{
        background: linear-gradient(
            90deg,
            rgba(20,184,166,0.20),
            rgba(20,184,166,0.04)
        ) !important;
        color: {TEXT_HEAD} !important;
        border-left: 3px solid {ACCENT} !important;
    }}

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label input {{
        display: none;
    }}


    /* Sidebar Model Card */

    .model-card {{
        background: linear-gradient(
            145deg,
            #111C30,
            #0D1628
        );
        border: 1px solid {GRID};
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }}

    .model-card-title {{
        color: {ACCENT_2};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        margin-bottom: 0.45rem;
        font-weight: 650;
    }}

    .model-card-name {{
        color: {TEXT_HEAD};
        font-size: 1rem;
        font-weight: 650;
        margin-bottom: 0.9rem;
    }}

    .model-stat {{
        color: {MUTED};
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    .model-stat-value {{
        color: {TEXT_HEAD};
        font-size: 1.05rem;
        font-weight: 700;
    }}


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {{
        background: linear-gradient(
            145deg,
            #162238,
            #111C30
        );
        border: 1px solid {GRID};
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        min-height: 105px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        border-color: {GRID_FAINT};
    }}

    .metric-card::before {{
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: {ACCENT};
    }}

    .metric-card .label {{
        color: {MUTED};
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 600;
    }}

    .metric-card .value {{
        color: {TEXT_HEAD};
        font-size: 1.9rem;
        font-weight: 750;
        margin-top: 0.35rem;
    }}


    /* ========================================================
       SECTION / INFORMATION CARDS
       ======================================================== */

    .info-card {{
        background: #111C30;
        border: 1px solid {GRID};
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}

    .section-label {{
        color: {ACCENT_2};
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-weight: 650;
        margin-bottom: 0.55rem;
    }}

    .page-subtitle {{
        color: {MUTED};
        font-size: 1.03rem;
        line-height: 1.7;
        margin-bottom: 1.5rem;
        max-width: 1050px;
    }}


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button[kind="primary"] {{
        background: linear-gradient(
            135deg,
            {ACCENT},
            #0D9488
        ) !important;
        border: none !important;
        color: #031E1B !important;
        font-weight: 700 !important;
        border-radius: 9px !important;
        padding: 0.65rem 1.5rem !important;
        transition: all 0.2s ease;
    }}

    .stButton > button[kind="primary"]:hover {{
        background: {ACCENT_2} !important;
        color: #031E1B !important;
        transform: translateY(-1px);
    }}


    /* ========================================================
       INPUTS
       ======================================================== */

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {{
        background-color: {PANEL_INPUT} !important;
        border-color: {GRID} !important;
        border-radius: 9px !important;
    }}

    input {{
        color: {TEXT_HEAD} !important;
    }}


    /* ========================================================
       SLIDERS
       ======================================================== */

    [data-testid="stSlider"] [role="slider"] {{
        background-color: {ACCENT_2} !important;
        border-color: {ACCENT_2} !important;
    }}

    [data-testid="stSlider"] div[data-baseweb="slider"] > div > div {{
        background: linear-gradient(
            90deg,
            {ACCENT},
            {ACCENT_2}
        ) !important;
    }}


    /* ========================================================
       RISK BADGES
       ======================================================== */

    .risk-badge {{
        display: inline-block;
        padding: 0.65rem 1.1rem;
        border-radius: 9px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }}

    .risk-high {{
        background: rgba(244,63,94,0.13);
        color: #FDA4AF;
        border: 1px solid rgba(244,63,94,0.45);
    }}

    .risk-low {{
        background: rgba(16,185,129,0.13);
        color: #6EE7B7;
        border: 1px solid rgba(16,185,129,0.45);
    }}


    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {{
        border: 1px solid {GRID};
        border-radius: 10px;
        overflow: hidden;
    }}


    /* ========================================================
       EXPANDERS
       ======================================================== */

    [data-testid="stExpander"] {{
        background-color: {PANEL};
        border: 1px solid {GRID};
        border-radius: 10px;
    }}


    /* ========================================================
       CHECKBOXES & RADIO
       ======================================================== */

    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {{
        color: {TEXT} !important;
    }}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(label, value):
    """Render a styled metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title, subtitle):
    """Render consistent page title and subtitle."""
    st.title(title)
    st.markdown(
        f"""
        <div class="page-subtitle">
            {subtitle}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text):
    """Render uppercase section label."""
    st.markdown(
        f"""
        <div class="section-label">{text}</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model_final.pkl")
    features = joblib.load("models/model_features.pkl")
    threshold = joblib.load("models/decision_threshold.pkl")
    return model, features, threshold


@st.cache_data
def load_reference_data():
    df = pd.read_csv("data/European_Bank.csv")
    return df


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


model, FEATURES, THRESHOLD = load_artifacts()
raw_df = load_reference_data()
explainer = get_shap_explainer(model)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_feature_row(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_cr_card,
    is_active,
    salary,
):
    """Build a single-row dataframe matching exact training feature order."""
    balance_per_product = balance / max(num_products, 1)

    row = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_cr_card),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": salary,
        "Balance_per_Product": balance_per_product,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
    }

    return pd.DataFrame([row])[FEATURES]


def predict_row(row_df):
    proba = model.predict_proba(row_df)[:, 1][0]
    flag = int(proba >= THRESHOLD)
    return proba, flag


# ============================================================
# SCORE COMPLETE REFERENCE DATASET
# ============================================================

@st.cache_data
def score_reference_data(_features_tuple):
    df = raw_df.copy()

    df["Balance_per_Product"] = (
        df["Balance"] /
        df["NumOfProducts"].replace(0, 1)
    )

    df["Geography_Germany"] = (
        df["Geography"] == "Germany"
    ).astype(int)

    df["Geography_Spain"] = (
        df["Geography"] == "Spain"
    ).astype(int)

    df["Gender_Male"] = (
        df["Gender"] == "Male"
    ).astype(int)

    X = df[list(_features_tuple)]

    df["churn_probability"] = model.predict_proba(X)[:, 1]

    df["predicted_churn"] = (
        df["churn_probability"] >= THRESHOLD
    ).astype(int)

    return df


scored_df = score_reference_data(tuple(FEATURES))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="brand-white">🏦 Churn Risk</span>
            <span class="brand-teal">Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🧮 Churn Risk Calculator",
            "📈 Probability Distribution",
            "🔍 Feature Importance",
            "🎛️ What-If Simulator",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="model-card">
            <div class="model-card-title">MODEL</div>
            <div class="model-card-name">⚡ LightGBM (Tuned)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sb1, sb2 = st.columns(2)

    with sb1:
        st.markdown(
            f"""
            <div class="model-stat">TEST ROC-AUC</div>
            <div class="model-stat-value">0.8699</div>
            """,
            unsafe_allow_html=True,
        )

    with sb2:
        st.markdown(
            f"""
            <div class="model-stat">THRESHOLD</div>
            <div class="model-stat-value">{THRESHOLD:.3f}</div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "📊 Overview":

    page_header(
        "Bank Customer Churn — Risk Intelligence",
        "This dashboard assigns churn risk probabilities to customers before they leave, enabling proactive retention rather than reactive damage control."
    )

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Customers",
            f"{len(raw_df):,}"
        )

    with c2:
        metric_card(
            "Historical Churn Rate",
            f"{raw_df['Exited'].mean()*100:.1f}%"
        )

    with c3:
        metric_card(
            "Model ROC-AUC",
            "0.8699"
        )

    with c4:
        metric_card(
            "High-Risk Flagged",
            f"{scored_df['predicted_churn'].sum():,}"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### How to read this dashboard")

    # Dedented HTML block to prevent Markdown from interpreting spaces as code blocks
    st.markdown(
        """
<div class="info-card">
    <div style="margin-bottom:16px;">
        <span style="color:#22D3EE;font-weight:700;">🧮 Churn Risk Calculator</span>
        <span style="color:#94A3B8;"> — Score an individual customer's likelihood of leaving.</span>
    </div>
    <div style="margin-bottom:16px;">
        <span style="color:#22D3EE;font-weight:700;">📈 Probability Distribution</span>
        <span style="color:#94A3B8;"> — Understand how predicted risk is distributed across customers.</span>
    </div>
    <div style="margin-bottom:16px;">
        <span style="color:#A78BFA;font-weight:700;">🔍 Feature Importance</span>
        <span style="color:#94A3B8;"> — Discover which customer attributes influence predictions.</span>
    </div>
    <div>
        <span style="color:#F59E0B;font-weight:700;">🎛️ What-If Simulator</span>
        <span style="color:#94A3B8;"> — Test how changing customer attributes affects churn probability.</span>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Highest-risk customers")

    top10 = (
        scored_df[
            [
                "CreditScore",
                "Geography",
                "Gender",
                "Age",
                "Balance",
                "NumOfProducts",
                "churn_probability",
                "predicted_churn",
            ]
        ]
        .sort_values("churn_probability", ascending=False)
        .head(10)
    )

    st.dataframe(
        top10.style.format(
            {
                "churn_probability": "{:.1%}",
                "Balance": "{:,.0f}",
            }
        ).background_gradient(
            subset=["churn_probability"],
            cmap="Reds",
        ),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# PAGE 2 — CHURN RISK CALCULATOR
# ============================================================

elif page == "🧮 Churn Risk Calculator":

    page_header(
        "Churn Risk Calculator",
        "Enter a customer's details to estimate their probability of churning."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        section_label("CUSTOMER PROFILE")
        credit_score = st.slider("Credit Score", 350, 850, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])

    with col2:
        section_label("TENURE & BALANCE")
        age = st.slider("Age", 18, 92, 38)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
        balance = st.number_input("Account Balance", min_value=0.0, max_value=260000.0, value=97000.0, step=1000.0)

    with col3:
        section_label("ENGAGEMENT")
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.checkbox("Has Credit Card", value=True)
        is_active = st.checkbox("Is Active Member", value=True)
        salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=100000.0, step=1000.0)

    st.write("")

    calc = st.button("Calculate Churn Risk", type="primary", width="content")

    if calc:
        row = build_feature_row(
            credit_score, geography, gender, age, tenure, balance, num_products, has_cr_card, is_active, salary
        )
        proba, flag = predict_row(row)

        st.markdown("---")
        section_label("PREDICTION RESULT")
        st.markdown("### Risk Assessment")

        r1, r2 = st.columns([1, 2])

        with r1:
            metric_card("Churn Probability", f"{proba:.1%}")
            badge_class = "risk-high" if flag else "risk-low"
            badge_text = (
                f"⚠️ HIGH RISK — Above threshold ({THRESHOLD:.1%})"
                if flag
                else f"✅ LOW RISK — Below threshold ({THRESHOLD:.1%})"
            )
            st.markdown(
                f"""
                <div class="risk-badge {badge_class}">
                    {badge_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            fig, ax = plt.subplots(figsize=(7, 1.5))
            ax.barh([0], [1], color=GRID, height=0.5, zorder=1)
            ax.barh([0], [proba], color=RISK if flag else GOOD, height=0.5, zorder=2)
            ax.axvline(THRESHOLD, color=TEXT, linestyle="--", linewidth=1.2, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Churn probability")
            ax.grid(False)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        with st.expander("Why this prediction? — SHAP Explanation"):
            shap_vals = explainer.shap_values(row)
            shap_vals_churn = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            vals = shap_vals_churn[0]
            order = np.argsort(np.abs(vals))
            colors = [RISK if v > 0 else ACCENT_2 for v in vals[order]]

            ax2.barh(np.array(FEATURES)[order], vals[order], color=colors)
            ax2.axvline(0, color=MUTED, linewidth=0.8)
            ax2.set_xlabel("SHAP value (impact on churn probability)")
            fig2.tight_layout()
            st.pyplot(fig2, width="stretch")
            plt.close(fig2)


# ============================================================
# PAGE 3 — PROBABILITY DISTRIBUTION
# ============================================================

elif page == "📈 Probability Distribution":

    page_header(
        "Churn Probability Distribution",
        "Explore how predicted churn risk is distributed across the entire customer base."
    )

    st.markdown("### Risk Distribution Across Customers")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(
        scored_df.loc[scored_df["Exited"] == 0, "churn_probability"],
        bins=40, alpha=0.75, label="Actually retained", color=GOOD,
    )
    ax.hist(
        scored_df.loc[scored_df["Exited"] == 1, "churn_probability"],
        bins=40, alpha=0.75, label="Actually churned", color=RISK,
    )
    ax.axvline(THRESHOLD, color=TEXT, linestyle="--", linewidth=1.2, label=f"Decision threshold ({THRESHOLD:.2f})")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Number of customers")
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("### Segment Breakdown")
    seg1, seg2 = st.columns(2)

    with seg1:
        section_label("BY GEOGRAPHY")
        geo_avg = scored_df.groupby("Geography")["churn_probability"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.barh(geo_avg.index, geo_avg.values, color=ACCENT_2)
        ax.set_xlabel("Average churn probability")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with seg2:
        section_label("BY NUMBER OF PRODUCTS")
        prod_avg = scored_df.groupby("NumOfProducts")["churn_probability"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(prod_avg.index.astype(str), prod_avg.values, color=PURPLE)
        ax.set_xlabel("Number of Products")
        ax.set_ylabel("Average churn probability")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("### Risk Tier Breakdown")
    tier_counts = scored_df["predicted_churn"].value_counts().rename({0: "Below threshold", 1: "Above threshold"})

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(tier_counts.index, tier_counts.values, color=[GOOD, RISK])
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ============================================================
# PAGE 4 — FEATURE IMPORTANCE
# ============================================================

elif page == "🔍 Feature Importance":

    page_header(
        "Feature Importance Dashboard",
        "Understand which customer attributes drive the model's churn predictions and influence risk scores."
    )

    st.markdown("### Model Feature Importance")
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    colors = [ACCENT_2 if value > importances.median() else ACCENT for value in importances.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importances.index, importances.values, color=colors)
    ax.set_xlabel("Importance (split count)")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("### SHAP Summary — Impact and Direction")
    st.caption(
        "Red indicates higher feature values and blue indicates lower values. "
        "Features positioned toward the right increase predicted churn risk."
    )

    sample = scored_df[FEATURES].sample(min(1000, len(scored_df)), random_state=42)
    shap_vals = explainer.shap_values(sample)
    shap_vals_churn = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_vals_churn, sample, show=False)
    fig2 = plt.gcf()
    fig2.patch.set_facecolor(BG)

    for ax_ in fig2.get_axes():
        ax_.set_facecolor(BG)
        ax_.tick_params(colors=TEXT)
        ax_.xaxis.label.set_color(TEXT)
        ax_.yaxis.label.set_color(TEXT)

    fig2.tight_layout()
    st.pyplot(fig2, width="stretch")
    plt.close("all")


# ============================================================
# PAGE 5 — WHAT-IF SIMULATOR
# ============================================================

elif page == "🎛️ What-If Simulator":

    page_header(
        "What-If Scenario Simulator",
        "Adjust customer attributes and observe how changes in engagement, tenure, and financial profile affect churn probability."
    )

    section_label("STEP 1 — CHOOSE A STARTING PROFILE")

    preset = st.radio(
        "Start from",
        ["Typical customer (dataset median)", "Custom"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if preset == "Typical customer (dataset median)":
        base = dict(
            credit_score=int(raw_df["CreditScore"].median()),
            geography="France",
            gender="Female",
            age=int(raw_df["Age"].median()),
            tenure=int(raw_df["Tenure"].median()),
            balance=float(raw_df["Balance"].median()),
            num_products=1,
            has_cr_card=True,
            is_active=True,
            salary=float(raw_df["EstimatedSalary"].median()),
        )
    else:
        base = dict(
            credit_score=650,
            geography="France",
            gender="Female",
            age=38,
            tenure=5,
            balance=97000.0,
            num_products=1,
            has_cr_card=True,
            is_active=True,
            salary=100000.0,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    section_label("STEP 2 — ADJUST AND OBSERVE")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.markdown("### Engagement")
        sim_active = st.select_slider("Active member?", options=["No", "Yes"], value="Yes" if base["is_active"] else "No")
        sim_products = st.slider("Number of Products", 1, 4, base["num_products"])
        sim_tenure = st.slider("Tenure (years)", 0, 10, base["tenure"])

    with sim_col2:
        st.markdown("### Financial Profile")
        sim_balance = st.slider("Balance", 0, 260000, int(base["balance"]), step=5000)
        sim_age = st.slider("Age", 18, 92, base["age"])
        sim_credit = st.slider("Credit Score", 350, 850, base["credit_score"])

    row = build_feature_row(
        sim_credit,
        base["geography"],
        base["gender"],
        sim_age,
        sim_tenure,
        sim_balance,
        sim_products,
        base["has_cr_card"],
        sim_active == "Yes",
        base["salary"],
    )

    proba, flag = predict_row(row)

    st.markdown("---")
    section_label("STEP 3 — SIMULATION RESULT")
    st.markdown("### Risk Assessment")

    r1, r2 = st.columns([1, 2])

    with r1:
        metric_card("Simulated Churn Probability", f"{proba:.1%}")
        badge_class = "risk-high" if flag else "risk-low"
        badge_text = "⚠️ HIGH RISK" if flag else "✅ LOW RISK"
        st.markdown(
            f"""
            <div class="risk-badge {badge_class}">
                {badge_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r2:
        sweep_vals = list(range(1, 5))
        sweep_probs = []

        for p in sweep_vals:
            r = build_feature_row(
                sim_credit,
                base["geography"],
                base["gender"],
                sim_age,
                sim_tenure,
                sim_balance,
                p,
                base["has_cr_card"],
                sim_active == "Yes",
                base["salary"],
            )
            sweep_probs.append(model.predict_proba(r)[:, 1][0])

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(sweep_vals, sweep_probs, marker="o", color=ACCENT_2, linewidth=2.2, markersize=7)
        ax.axhline(THRESHOLD, color=WARN, linestyle="--", linewidth=1.2, label="Decision threshold")
        ax.set_xlabel("Number of Products")
        ax.set_ylabel("Churn probability")
        ax.set_title("Sensitivity to Number of Products", color=TEXT_HEAD)
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    # Dedented HTML block for What-If Simulator interpretation
    st.markdown(
        """
<div class="info-card">
    <div class="section-label">INTERPRETATION</div>
    <p style="color:#CBD5E1; line-height:1.7;">
    This simulation keeps the selected customer's other attributes
    fixed while changing the number of products. The resulting curve
    shows how product engagement may influence predicted churn risk
    according to the trained LightGBM model.
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )