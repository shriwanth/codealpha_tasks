import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)
from imblearn.over_sampling import SMOTE

def train_models():
    # Load data
    df = pd.read_csv('data/multi_disease_dataset.csv')
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Preprocessing - Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE (as requested, though data is balanced)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Models to train
    models = {
        "Logistic Regression": LogisticRegression(multi_class='multinomial', max_iter=1000),
        "SVM": SVC(probability=True, kernel='rbf'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    
    results = {}
    best_f1 = 0
    best_model_name = ""
    best_model = None
    
    os.makedirs('reports/figures', exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC-AUC for multi-class
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }
        
        print(f"{name} - F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    # Save results to a report
    results_df = pd.DataFrame(results).T
    results_df.to_csv('reports/model_comparison.csv')
    print("\nModel Comparison saved to reports/model_comparison.csv")
    
    # Save best model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print(f"Best model ({best_model_name}) saved to models/best_model.pkl")
    
    # --- Visualizations ---
    
    # 1. Confusion Matrix for best model
    y_pred_best = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(df.iloc[:, :-1].corr(), annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('reports/figures/correlation_heatmap.png')
    plt.close()
    
    # 3. Disease Distribution
    plt.figure(figsize=(8, 6))
    df['Target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Disease Class Distribution')
    plt.savefig('reports/figures/distribution.png')
    plt.close()
    
    # 4. Feature Importance (for RF or XGBoost)
    if hasattr(best_model, 'feature_importances_'):
        feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
        plt.figure(figsize=(10, 8))
        feat_importances.nlargest(15).plot(kind='barh', color='teal')
        plt.title(f'Top 15 Feature Importances ({best_model_name})')
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance.png')
        plt.close()

    print("Visualizations saved to reports/figures/")

if __name__ == "__main__":
    train_models()
