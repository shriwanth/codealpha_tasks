# 🏦 Credit Scoring Model

> A complete Machine Learning project to predict customer creditworthiness using historical financial data.

---

## 📋 Project Overview

This project implements an end-to-end Machine Learning pipeline that classifies customers as **Creditworthy (1)** or **Not Creditworthy (0)** based on their financial profile. It covers every stage of the ML lifecycle — from synthetic data generation and exploratory analysis, through feature engineering and model training, to evaluation, visualization, and deployment-ready inference.

---

## ❓ Problem Statement

Financial institutions need reliable, automated tools to assess credit risk before approving loans. Manual evaluation is slow, inconsistent, and prone to bias. By training classification models on historical customer data, we can build a consistent, explainable scoring system that helps lenders make data-driven decisions quickly.

---

## 📂 Project Structure

```
credit score/
│
├── credit_data.csv               # Synthetic dataset (500 records)
├── credit_scoring_model.py       # Main ML pipeline script
├── credit_scoring_model.pkl      # Best trained model (saved by joblib)
├── scaler.pkl                    # Feature scaler (StandardScaler)
├── README.md                     # This file
│
└── plots/
    ├── histograms.png            # Feature distribution histograms
    ├── correlation_heatmap.png   # Pearson correlation heatmap
    ├── feature_by_status.png     # Box plots by Credit Status
    ├── confusion_matrices.png    # Confusion matrices for all models
    ├── roc_curves.png            # ROC curves for all models
    ├── feature_importance.png    # Random Forest feature importance
    └── model_comparison.png     # Side-by-side metric comparison
```

---

## 🗃️ Dataset Description

The synthetic dataset (`credit_data.csv`) contains **500 records** with the following columns:

| Column            | Type    | Range           | Description                        |
|-------------------|---------|------------------|------------------------------------|
| `Age`             | Integer | 21 – 65          | Customer age in years              |
| `Income`          | Integer | 20,000 – 150,000 | Annual income (USD)                |
| `Debt`            | Integer | 1,000 – 80,000   | Total outstanding debt (USD)       |
| `LoanAmount`      | Integer | 5,000 – 100,000  | Requested loan amount (USD)        |
| `EmploymentYears` | Integer | 0 – 30           | Years of stable employment         |
| `Credit_Status`   | Binary  | 0 or 1           | **Target**: 1 = Creditworthy, 0 = Not |

### Target Assignment Logic

`Credit_Status` is assigned using a composite score:

```
score = 0.40 × (1 − debt_to_income)
      + 0.35 × (1 − loan_to_income / 2)
      + 0.25 × (employment_years / 30)
      + Gaussian noise (μ=0, σ=0.08)

Credit_Status = 1  if  score ≥ 0.50
```

---

## ⚙️ Feature Engineering

Two derived features are created to capture financial ratios, which are strong predictors of credit risk:

| Feature             | Formula                 | Interpretation                        |
|---------------------|-------------------------|---------------------------------------|
| `Debt_Income_Ratio` | `Debt / Income`         | Higher → more financially stressed    |
| `Loan_Income_Ratio` | `LoanAmount / Income`   | Higher → riskier loan request         |

Training features (7 total):
`Age`, `Income`, `Debt`, `LoanAmount`, `EmploymentYears`, `Debt_Income_Ratio`, `Loan_Income_Ratio`

---

## 🤖 Algorithms Used

| Model                  | Key Hyperparameters                                |
|------------------------|----------------------------------------------------|
| **Logistic Regression**| `max_iter=1000`                                   |
| **Decision Tree**      | `max_depth=6`, `min_samples_leaf=5`               |
| **Random Forest**      | `n_estimators=200`, `max_depth=10`, `n_jobs=-1`   |

All models use **StandardScaler** for feature normalization before training.

---

## 📊 Evaluation Metrics

Each model is evaluated on a **20% hold-out test set** using:

| Metric        | Description                                                |
|---------------|------------------------------------------------------------|
| **Accuracy**  | Overall fraction of correct predictions                    |
| **Precision** | Of predicted creditworthy, how many truly are?             |
| **Recall**    | Of actual creditworthy customers, how many were found?     |
| **F1 Score**  | Harmonic mean of Precision & Recall                        |
| **ROC-AUC**   | Area Under the ROC Curve (discrimination ability)          |

---

## 📈 Results

> Results are printed to the console after running the pipeline. Typical expected values:

| Model               | Accuracy  | Precision | Recall    | F1 Score  | ROC-AUC   |
|---------------------|-----------|-----------|-----------|-----------|-----------|
| Logistic Regression | ~0.79     | ~0.79     | ~0.83     | ~0.81     | ~0.86     |
| Decision Tree       | ~0.77     | ~0.78     | ~0.81     | ~0.79     | ~0.83     |
| **Random Forest**   | **~0.84** | **~0.84** | **~0.87** | **~0.85** | **~0.91** |

> **Best model**: Random Forest (highest ROC-AUC + F1 composite score)

---

## 🚀 How to Run the Project

### 1. Prerequisites

Ensure Python 3.8+ is installed, then install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### 2. Run the Pipeline

```bash
cd "credit score"
python credit_scoring_model.py
```

Running the script will:
1. Generate `credit_data.csv` (if not already present)
2. Perform preprocessing and EDA
3. Engineer features
4. Train all three models
5. Evaluate and compare performance
6. Save plots to `plots/`
7. Save the best model as `credit_scoring_model.pkl`
8. Save the scaler as `scaler.pkl`
9. Run three demo predictions

### 3. Make a Custom Prediction

```python
from credit_scoring_model import predict_creditworthiness

result = predict_creditworthiness(
    age=35,
    income=80_000,
    debt=10_000,
    loan_amount=20_000,
    employment_years=8,
)
print(result)   # "Creditworthy" or "Not Creditworthy"
```

---

## 📦 Dependencies

| Library      | Purpose                                      |
|--------------|----------------------------------------------|
| `numpy`      | Numerical computations                       |
| `pandas`     | Data loading and manipulation                |
| `matplotlib` | Plotting                                     |
| `seaborn`    | Statistical visualizations                   |
| `scikit-learn` | ML models, preprocessing, evaluation      |
| `joblib`     | Model serialization / deserialization        |

---

## 🔮 Future Improvements

1. **Real Dataset** – Replace synthetic data with a real credit dataset (e.g., UCI German Credit, Kaggle Give Me Some Credit).
2. **Hyperparameter Tuning** – Use `GridSearchCV` or `Optuna` to optimize model parameters.
3. **XGBoost / LightGBM** – Experiment with gradient boosting algorithms for better performance.
4. **SHAP Explainability** – Add SHAP value analysis for model interpretability.
5. **Class Imbalance Handling** – Apply SMOTE or class-weight adjustments for imbalanced real-world data.
6. **Cross-Validation** – Use k-fold CV for more robust evaluation.
7. **Web API** – Wrap the prediction function in a FastAPI/Flask endpoint.
8. **Pipeline Object** – Bundle scaler + model into a single `sklearn.Pipeline` for simpler deployment.
9. **Threshold Tuning** – Optimize the decision threshold using F1/Youden-J score for business objectives.
10. **MLflow Tracking** – Log experiments, parameters, and metrics with MLflow.

---

## 👤 Author

Built with ❤️ using Python, Scikit-Learn, and Matplotlib.

---

*Last updated: 2026-06-22*
