"""
=======================================================================
  Credit Scoring Model  –  Complete Machine Learning Pipeline
  Author : Antigravity AI
  Date   : 2026-06-22
=======================================================================

Objective
---------
Predict whether a customer is creditworthy (1) or not creditworthy (0)
using historical financial data.

Deliverables
------------
  • credit_data.csv               – Synthetic dataset (500 records)
  • credit_scoring_model.pkl      – Best trained model
  • scaler.pkl                    – Feature scaler
  • credit_scoring_model.py       – This script
  • README.md                     – Project documentation
"""

# ─────────────────────────────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend (safe in all envs)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve,
)

warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────
# 1.  Synthetic Dataset Generation
# ─────────────────────────────────────────────────────────────────────
def generate_dataset(n_samples: int = 500, filepath: str = "credit_data.csv") -> pd.DataFrame:
    """
    Generate a synthetic credit dataset with logical target assignment.

    Credit_Status = 1 (Creditworthy) when the customer shows:
        • Higher income relative to debt (low debt-to-income ratio)
        • Stable employment (higher EmploymentYears)
        • Manageable loan relative to income
    """
    print("\n" + "="*60)
    print("  STEP 1 – Generating Synthetic Dataset")
    print("="*60)

    age             = np.random.randint(21, 66,       size=n_samples)
    income          = np.random.randint(20_000, 150_001, size=n_samples)
    debt            = np.random.randint(1_000, 80_001,  size=n_samples)
    loan_amount     = np.random.randint(5_000, 100_001, size=n_samples)
    employment_years = np.random.randint(0, 31,        size=n_samples)

    # ── Logical creditworthiness score  ──────────────────────────────
    debt_to_income  = debt  / income                       # lower  → better
    loan_to_income  = loan_amount / income                 # lower  → better
    emp_score       = employment_years / 30                # higher → better

    # Composite score (0–1 scale, higher = more creditworthy)
    credit_score = (
        0.40 * (1 - np.clip(debt_to_income, 0, 1)) +
        0.35 * (1 - np.clip(loan_to_income, 0, 2) / 2) +
        0.25 * emp_score
    )

    # Add noise so the boundary is not perfectly clean
    noise = np.random.normal(0, 0.08, n_samples)
    credit_score = np.clip(credit_score + noise, 0, 1)

    credit_status = (credit_score >= 0.50).astype(int)

    df = pd.DataFrame({
        "Age"             : age,
        "Income"          : income,
        "Debt"            : debt,
        "LoanAmount"      : loan_amount,
        "EmploymentYears" : employment_years,
        "Credit_Status"   : credit_status,
    })

    df.to_csv(filepath, index=False)
    print(f"  ✔ Dataset created  →  {filepath}  ({n_samples} records)")
    print(f"  ✔ Class distribution:\n{df['Credit_Status'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────
# 2.  Data Loading
# ─────────────────────────────────────────────────────────────────────
def load_data(filepath: str = "credit_data.csv") -> pd.DataFrame:
    """Load the credit dataset from a CSV file."""
    print("\n" + "="*60)
    print("  STEP 2 – Data Loading")
    print("="*60)

    df = pd.read_csv(filepath)
    print(f"  ✔ Loaded  →  shape: {df.shape}")
    print("\n  First 5 rows:")
    print(df.head().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────
# 3.  Data Preprocessing
# ─────────────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Check & handle missing values
    • Encode categorical features
    • Display dataset statistics
    """
    print("\n" + "="*60)
    print("  STEP 3 – Data Preprocessing")
    print("="*60)

    # Missing values
    missing = df.isnull().sum()
    print("\n  Missing values per column:")
    print(missing.to_string())

    if missing.sum() > 0:
        # Numerical columns → fill with median
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        print("  ✔ Missing values imputed with column median.")
    else:
        print("  ✔ No missing values found.")

    # Categorical encoding (none in our dataset; kept for extensibility)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        print(f"\n  Encoding categorical columns: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print("  ✔ One-hot encoding applied.")

    # Statistics
    print("\n  Dataset Statistics:")
    print(df.describe().to_string())

    return df


# ─────────────────────────────────────────────────────────────────────
# 4.  Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────────────────────────────
def perform_eda(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    • Dataset shape & class distribution
    • Histograms for each feature
    • Correlation heatmap
    • Feature distributions by Credit_Status
    """
    print("\n" + "="*60)
    print("  STEP 4 – Exploratory Data Analysis (EDA)")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # ── Shape & class distribution ────────────────────────────────────
    print(f"\n  Dataset shape  : {df.shape}")
    print(f"\n  Class distribution:\n{df['Credit_Status'].value_counts().to_string()}")
    print(f"  Class balance  : {df['Credit_Status'].value_counts(normalize=True).round(3).to_string()}")

    features = ["Age", "Income", "Debt", "LoanAmount", "EmploymentYears"]

    # ── Histograms ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold")
    for i, col in enumerate(features):
        ax = axes[i // 3][i % 3]
        ax.hist(df[col], bins=25, color="#4C6EF5", edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.3)
    # Last subplot → class distribution bar
    ax = axes[1][2]
    counts = df["Credit_Status"].value_counts()
    ax.bar(["Not Creditworthy (0)", "Creditworthy (1)"],
           counts.values,
           color=["#FA5252", "#51CF66"],
           edgecolor="white")
    ax.set_title("Credit Status Distribution", fontsize=12)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "histograms.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✔ Histograms saved  →  {hist_path}")

    # ── Correlation heatmap ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5,
        annot_kws={"size": 10}, ax=ax,
    )
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Heatmap saved     →  {heatmap_path}")

    # ── Feature distributions by Credit_Status (box plots) ────────────
    fig, axes = plt.subplots(1, len(features), figsize=(18, 5))
    fig.suptitle("Feature Distributions by Credit Status", fontsize=14, fontweight="bold")
    box_colors = ["#FA5252", "#51CF66"]
    for i, col in enumerate(features):
        ax = axes[i]
        groups = [df.loc[df["Credit_Status"] == lbl, col].values for lbl in [0, 1]]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Not CW (0)", "CW (1)"])
        ax.set_title(col, fontsize=12)
        ax.set_xlabel("Credit Status")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "feature_by_status.png")
    fig.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Box plots saved   →  {boxplot_path}")


# ─────────────────────────────────────────────────────────────────────
# 5.  Feature Engineering
# ─────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    """
    Create derived financial ratios:
      • Debt_Income_Ratio  = Debt / Income
      • Loan_Income_Ratio  = LoanAmount / Income
    Returns (X, y) split with all features.
    """
    print("\n" + "="*60)
    print("  STEP 5 – Feature Engineering")
    print("="*60)

    df = df.copy()
    df["Debt_Income_Ratio"] = df["Debt"] / df["Income"]
    df["Loan_Income_Ratio"] = df["LoanAmount"] / df["Income"]

    print("  ✔ Added  Debt_Income_Ratio  = Debt / Income")
    print("  ✔ Added  Loan_Income_Ratio  = LoanAmount / Income")

    feature_cols = [
        "Age", "Income", "Debt", "LoanAmount", "EmploymentYears",
        "Debt_Income_Ratio", "Loan_Income_Ratio",
    ]
    target_col = "Credit_Status"

    X = df[feature_cols]
    y = df[target_col]

    print(f"\n  Feature matrix shape : {X.shape}")
    print(f"  Target vector shape  : {y.shape}")
    print(f"  Features used        : {feature_cols}")

    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────
# 6.  Model Building & Training
# ─────────────────────────────────────────────────────────────────────
def build_and_train_models(X_train, y_train):
    """
    Instantiate and train three classifiers:
      1. Logistic Regression
      2. Decision Tree
      3. Random Forest
    Returns a dict of fitted models.
    """
    print("\n" + "="*60)
    print("  STEP 6 – Model Building & Training")
    print("="*60)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  ✔ Trained  →  {name}")

    return models


# ─────────────────────────────────────────────────────────────────────
# 7.  Model Evaluation
# ─────────────────────────────────────────────────────────────────────
def evaluate_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Compute Accuracy, Precision, Recall, F1, and ROC-AUC for each model.
    Returns a summary DataFrame.
    """
    print("\n" + "="*60)
    print("  STEP 7 – Model Evaluation")
    print("="*60)

    results = []
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model"    : name,
            "Accuracy" : round(accuracy_score(y_test, y_pred),       4),
            "Precision": round(precision_score(y_test, y_pred),      4),
            "Recall"   : round(recall_score(y_test, y_pred),         4),
            "F1 Score" : round(f1_score(y_test, y_pred),             4),
            "ROC-AUC"  : round(roc_auc_score(y_test, y_proba),       4),
        }
        results.append(metrics)
        print(f"\n  ── {name} ──────────────────────────────")
        for k, v in metrics.items():
            if k != "Model":
                print(f"     {k:<12}: {v}")

    results_df = pd.DataFrame(results).set_index("Model")
    print("\n  ── Comparison Table ────────────────────────")
    print(results_df.to_string())
    return results_df


# ─────────────────────────────────────────────────────────────────────
# 8.  Visualization  (Confusion Matrix, ROC Curve, Feature Importance)
# ─────────────────────────────────────────────────────────────────────
def visualize_results(models: dict, X_test, y_test,
                      feature_cols: list, output_dir: str = "plots") -> None:
    """
    Plot per-model confusion matrices, combined ROC curves,
    and Random Forest feature importance.
    """
    print("\n" + "="*60)
    print("  STEP 8 – Visualization")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # ── Confusion matrices ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        disp   = ConfusionMatrixDisplay(cm, display_labels=["Not CW", "CW"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Confusion matrices →  {cm_path}")

    # ── ROC Curves ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4C6EF5", "#FA5252", "#51CF66"]
    for (name, model), color in zip(models.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ ROC curves         →  {roc_path}")

    # ── Feature Importance (Random Forest) ───────────────────────────
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("viridis", len(feature_cols))
    bars = ax.barh(
        [feature_cols[i] for i in sorted_idx[::-1]],
        importances[sorted_idx[::-1]],
        color=palette,
        edgecolor="white",
    )
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Random Forest – Feature Importance", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, importances[sorted_idx[::-1]]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Feature importance →  {fi_path}")

    # ── Model comparison bar chart ────────────────────────────────────
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    model_names     = list(models.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_colors = ["#4C6EF5", "#FA5252", "#51CF66"]
    for i, (name, color) in enumerate(zip(model_names, bar_colors)):
        model = models[name]
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        vals = [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, y_proba),
        ]
        ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85, edgecolor="white")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison – All Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    comp_path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Model comparison   →  {comp_path}")


# ─────────────────────────────────────────────────────────────────────
# 9.  Model Selection & Saving
# ─────────────────────────────────────────────────────────────────────
def select_and_save_best_model(models: dict, results_df: pd.DataFrame,
                                scaler: StandardScaler,
                                model_path : str = "credit_scoring_model.pkl",
                                scaler_path: str = "scaler.pkl") -> tuple:
    """
    Select the best model based on composite (ROC-AUC + F1) score.
    Save both the model and scaler with joblib.
    Returns (best_model_name, best_model).
    """
    print("\n" + "="*60)
    print("  STEP 9 – Model Selection & Saving")
    print("="*60)

    # Composite score
    results_df["Composite"] = (results_df["ROC-AUC"] + results_df["F1 Score"]) / 2
    best_name = results_df["Composite"].idxmax()
    best_model = models[best_name]

    print(f"\n  Best model selected  →  {best_name}")
    print(f"  ROC-AUC : {results_df.loc[best_name, 'ROC-AUC']:.4f}")
    print(f"  F1 Score: {results_df.loc[best_name, 'F1 Score']:.4f}")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler,     scaler_path)
    print(f"\n  ✔ Model saved  →  {model_path}")
    print(f"  ✔ Scaler saved →  {scaler_path}")

    return best_name, best_model


# ─────────────────────────────────────────────────────────────────────
# 10.  Prediction Module
# ─────────────────────────────────────────────────────────────────────
def predict_creditworthiness(
    age: float,
    income: float,
    debt: float,
    loan_amount: float,
    employment_years: float,
    model_path : str = "credit_scoring_model.pkl",
    scaler_path: str = "scaler.pkl",
) -> str:
    """
    Predict creditworthiness for a single customer.

    Parameters
    ----------
    age              : Customer age (21–65)
    income           : Annual income
    debt             : Total outstanding debt
    loan_amount      : Requested loan amount
    employment_years : Years of stable employment
    model_path       : Path to saved model  (.pkl)
    scaler_path      : Path to saved scaler (.pkl)

    Returns
    -------
    str  →  "Creditworthy" or "Not Creditworthy"
    """
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Feature engineering (must match training)
    debt_income_ratio = debt        / income
    loan_income_ratio = loan_amount / income

    features = np.array([[
        age, income, debt, loan_amount, employment_years,
        debt_income_ratio, loan_income_ratio,
    ]])

    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)[0]
    probability     = model.predict_proba(features_scaled)[0][1]

    label = "Creditworthy" if prediction == 1 else "Not Creditworthy"
    print(f"\n  Prediction  : {label}")
    print(f"  Probability : {probability:.4f}  (creditworthy)")
    return label


# ─────────────────────────────────────────────────────────────────────
# 11.  Main Pipeline
# ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   Credit Scoring Model  –  ML Pipeline" + " "*19 + "║")
    print("╚" + "═"*58 + "╝")

    # ── 1. Generate or Load dataset ───────────────────────────────────
    DATA_FILE = "credit_data.csv"
    if not os.path.exists(DATA_FILE):
        df = generate_dataset(n_samples=500, filepath=DATA_FILE)
    else:
        print(f"\n  ℹ  '{DATA_FILE}' already exists – skipping generation.")

    # ── 2. Load data ──────────────────────────────────────────────────
    df = load_data(DATA_FILE)

    # ── 3. Preprocess ─────────────────────────────────────────────────
    df = preprocess_data(df)

    # ── 4. EDA ───────────────────────────────────────────────────────
    perform_eda(df)

    # ── 5. Feature engineering ────────────────────────────────────────
    X, y, feature_cols = engineer_features(df)

    # ── Train / Test split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")

    # ── Feature Scaling ───────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("  ✔ Features scaled with StandardScaler.")

    # ── 6. Train models ───────────────────────────────────────────────
    models = build_and_train_models(X_train, y_train)

    # ── 7. Evaluate ───────────────────────────────────────────────────
    results_df = evaluate_models(models, X_test, y_test)

    # ── 8. Visualize ─────────────────────────────────────────────────
    visualize_results(models, X_test, y_test, feature_cols)

    # ── 9. Select & Save ─────────────────────────────────────────────
    best_name, best_model = select_and_save_best_model(
        models, results_df, scaler
    )

    # ── 10. Demo prediction ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 10 – Demo Predictions")
    print("="*60)

    demo_cases = [
        {"Age": 35, "Income": 80_000, "Debt": 10_000, "LoanAmount": 20_000,
         "EmploymentYears": 8,  "desc": "Likely  Creditworthy"},
        {"Age": 25, "Income": 22_000, "Debt": 60_000, "LoanAmount": 90_000,
         "EmploymentYears": 1,  "desc": "Likely  Not Creditworthy"},
        {"Age": 45, "Income": 120_000,"Debt": 5_000,  "LoanAmount": 30_000,
         "EmploymentYears": 20, "desc": "Very likely Creditworthy"},
    ]

    for case in demo_cases:
        print(f"\n  Customer profile: {case['desc']}")
        print(f"    Age={case['Age']}  Income={case['Income']:,}  Debt={case['Debt']:,}"
              f"  Loan={case['LoanAmount']:,}  EmpYrs={case['EmploymentYears']}")
        predict_creditworthiness(
            age=case["Age"],
            income=case["Income"],
            debt=case["Debt"],
            loan_amount=case["LoanAmount"],
            employment_years=case["EmploymentYears"],
        )

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   Pipeline complete !  All artefacts saved." + " "*13 + "║")
    print("╚" + "═"*58 + "╝\n")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
