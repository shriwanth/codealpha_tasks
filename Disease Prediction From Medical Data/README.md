# Multi-Disease Prediction System 🏥

An intelligent AI system powered by Machine Learning and SHAP explainability to predict:
- Heart Disease
- Diabetes
- Breast Cancer
- Or No Disease

## Features
- **Unified Input Form**: Single entry point for all medical parameters.
- **Explainable AI**: SHAP-based reasoning for every prediction.
- **Visual Analytics**: Interactive dashboards and performance metrics.
- **Medical Report**: Automated PDF generator for patient data and results.
- **Multi-Model Comparison**: Evaluates Logistic Regression, SVM, Random Forest, and XGBoost.

## Project Structure
- `data/`: Contains the synthetic healthcare dataset.
- `models/`: Trained model, scaler, and label encoder.
- `reports/`: Comparison metrics and visualization figures.
- `src/`: Source code for data prep, training, and explainability.
- `app.py`: Main Streamlit application.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data & Train Model** (Done):
   ```bash
   python src/data_preparation.py
   python src/train_model.py
   ```

3. **Launch Streamlit App**:
   ```bash
   streamlit run app.py
   ```

## Tech Stack
- **Languages**: Python (Pandas, NumPy)
- **ML**: Scikit-Learn, XGBoost, SHAP
- **UI**: Streamlit
- **Reports**: FPDF, Matplotlib, Seaborn
