import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_shap_explanation(patient_df):
    """
    Generate SHAP explanation for a single prediction.
    """
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le = joblib.load('models/label_encoder.pkl')
    
    # Scale user data
    patient_scaled = scaler.transform(patient_df)
    
    # Prediction
    pred_class_idx = model.predict(patient_scaled)[0]
    pred_class_name = le.inverse_transform([pred_class_idx])[0]
    
    # Explainability
    # For multi-class, SHAP provides a list of arrays (one for each class)
    # We take the explainer suitable for the model type
    if 'XGB' in str(type(model)) or 'RandomForest' in str(type(model)):
        explainer = shap.TreeExplainer(model)
    else:
        # For LogReg or SVM, use KernelExplainer or LinearExplainer
        # Since we use synthetic data info, KernelExplainer is safer
        # But KernelExplainer needs a background dataset.
        # We'll use a sample from the training data if needed.
        df = pd.read_csv('data/multi_disease_dataset.csv')
        X = df.drop('Target', axis=1)
        X_sample = scaler.transform(X.sample(100, random_state=42))
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)
    
    shap_values = explainer.shap_values(patient_scaled)
    
    # For multi-class TreeExplainer, shap_values might be a list or a 3D array
    # XGBoost multi-class returns a 3D array (samples, features, classes) or list
    if isinstance(shap_values, list):
        cur_shap_values = shap_values[pred_class_idx]
    else:
        # Depending on SHAP version and model, indices might vary
        if len(shap_values.shape) == 3:
            cur_shap_values = shap_values[0, :, pred_class_idx]
        else:
            cur_shap_values = shap_values[0]

    # Create a nice summary of top and bottom features
    importances = pd.DataFrame({
        'Feature': patient_df.columns,
        'SHAP Value': cur_shap_values.flatten()
    })
    importances['Absolute Value'] = importances['SHAP Value'].abs()
    importances = importances.sort_values(by='Absolute Value', ascending=False)
    
    return pred_class_name, importances, cur_shap_values

def plot_shap_summary(cur_shap_values, patient_df, class_name):
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.bar_plot(cur_shap_values.flatten(), feature_names=patient_df.columns, show=False)
    plt.title(f"Factors contributing to '{class_name}' Prediction")
    plt.tight_layout()
    plt.savefig('reports/figures/current_prediction_shap.png')
    plt.close()
    return 'reports/figures/current_prediction_shap.png'
