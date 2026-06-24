import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.explainability import get_shap_explanation, plot_shap_summary
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Multi-Disease AI Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        border: 1px solid #4caf50;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #1e2128;
        border-left: 5px solid #4caf50;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    h1, h2, h3 {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_assets():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le = joblib.load('models/label_encoder.pkl')
    return model, scaler, le

def create_pdf_report(patient_data, prediction, confidence, risk, shap_importances):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Medical Prediction Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Prediction: {prediction}", ln=True)
    pdf.cell(200, 10, f"Confidence Score: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, f"Risk Level: {risk}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Patient Data Provided:", ln=True)
    pdf.set_font("Arial", size=10)
    
    for col, val in patient_data.items():
        pdf.cell(100, 8, f"{col}: {val}", border=1)
        pdf.ln()

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Key Contributing Factors:", ln=True)
    pdf.set_font("Arial", size=10)
    for i, row in shap_importances.head(5).iterrows():
        pdf.cell(200, 8, f"{row['Feature']}: Impact {row['SHAP Value']:.4f}", ln=True)
    
    os.makedirs('temp', exist_ok=True)
    pdf_path = "temp/medical_report.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Main Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2854/2854332.png", width=100)
st.sidebar.title("AI Health Suite")
page = st.sidebar.selectbox("Navigation", ["Prediction Dashboard", "Risk Insights", "Model Analytics", "Patient History"])

model, scaler, le = load_assets()

if page == "Prediction Dashboard":
    st.title("🏥 Multi-Disease Intelligent Prediction System")
    st.markdown("Enter patient medical parameters across different categories for an automated disease screening.")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📋 Demographic & Basic")
            age = st.number_input("Age", 18, 100, 45)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
            bmi = st.number_input("BMI", 10.0, 60.0, 24.5)
            glucose = st.number_input("Glucose Level", 50, 300, 100)
            bp_sys = st.number_input("Blood Pressure (Systolic)", 80, 200, 120)
            
        with col2:
            st.subheader("❤️ Cardiovascular")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trtp = st.number_input("Resting Blood Pressure", 80, 200, 120)
            chol = st.number_input("Cholesterol", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
            restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1])
            oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.2, 1.0)

        with col3:
            st.subheader("🎗️ Breast Cancer & Diabetes")
            preg = st.number_input("Pregnancies", 0, 20, 0)
            skin = st.number_input("Skin Thickness", 0, 100, 20)
            insulin = st.number_input("Insulin", 0, 900, 80)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
            radius = st.number_input("Radius Mean", 5.0, 30.0, 14.0)
            texture = st.number_input("Texture Mean", 5.0, 40.0, 19.0)
            perimeter = st.number_input("Perimeter Mean", 40.0, 190.0, 90.0)
            area = st.number_input("Area Mean", 100.0, 2500.0, 650.0)
            smooth = st.number_input("Smoothness Mean", 0.0, 0.2, 0.1)
            compact = st.number_input("Compactness Mean", 0.0, 0.4, 0.1)
            concave = st.number_input("Concavity Mean", 0.0, 0.5, 0.1)

        submitted = st.form_submit_button("Analyze Patient Data")

    if submitted:
        # Prepare input data
        # Feature order must match training
        input_data = {
            'Age': age, 'Sex': sex, 'ChestPainType': cp, 'RestingBP': trtp, 
            'Cholesterol': chol, 'FastingBS': fbs, 'RestECG': restecg, 'MaxHR': thalach, 
            'ExerciseAngina': exang, 'Oldpeak': oldpeak, 'Pregnancies': preg, 
            'Glucose': glucose, 'BloodPressure': bp_sys, 'SkinThickness': skin, 
            'Insulin': insulin, 'BMI': bmi, 'DiabetesPedigreeFunction': dpf, 
            'Radius_mean': radius, 'Texture_mean': texture, 'Perimeter_mean': perimeter, 
            'Area_mean': area, 'Smoothness_mean': smooth, 'Compactness_mean': compact, 
            'Concavity_mean': concave
        }
        
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        # Prediction
        probs = model.predict_proba(input_scaled)[0]
        pred_idx = np.argmax(probs)
        prediction = le.classes_[pred_idx]
        confidence = probs[pred_idx] * 100
        
        # Risk Assessment
        risk = "Low"
        if confidence > 90 and prediction != "No Disease": risk = "High"
        elif confidence > 70 and prediction != "No Disease": risk = "Moderate"
        
        # Display Results
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Prediction Result</h3>
                <h1 style='color: {"#f44336" if prediction != "No Disease" else "#4caf50"}'>{prediction}</h1>
                <p>Confidence Score: <b>{confidence:.2f}%</b></p>
                <p>Risk Level: <b style='color: {"#f44336" if risk == "High" else "#ff9800" if risk == "Moderate" else "#4caf50"}'>{risk}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendation
            if prediction == "Heart Disease":
                st.warning("🚨 Recommendation: Consult a Cardiologist immediately. Monitor blood pressure and reduce sodium intake.")
            elif prediction == "Diabetes":
                st.warning("🩸 Recommendation: Consult an Endocrinologist. Check blood sugar levels daily and adjust diet.")
            elif prediction == "Breast Cancer":
                st.error("🎗️ Recommendation: Urgent Oncology consultation. Follow-up mammogram or biopsy advised.")
            else:
                st.success("✅ Recommendation: Patient appears healthy. Maintain regular exercise and balanced diet.")

        with res_col2:
            st.subheader("💡 Explainable AI (SHAP Reasoning)")
            # Get SHAP
            pred_name, importances, shap_v = get_shap_explanation(input_df)
            shap_fig_path = plot_shap_summary(shap_v, input_df, pred_name)
            st.image(shap_fig_path)
            
            # Top contributing factors text
            st.markdown("### Top 3 Contributing Factors:")
            for i, row in importances.head(3).iterrows():
                impact = "Increases Risk" if row['SHAP Value'] > 0 else "Decreases Risk"
                st.write(f"- **{row['Feature']}**: {input_data[row['Feature']]} (Impact: {row['SHAP Value']:.3f})")

        # Bonus: Download PDF
        pdf_report = create_pdf_report(input_data, prediction, confidence, risk, importances)
        with open(pdf_report, "rb") as f:
            st.download_button("📥 Download Detailed Medical Report", f, "Medical_Report.pdf", "application/pdf")

elif page == "Model Analytics":
    st.title("📊 Model Performance & Insights")
    
    if os.path.exists('reports/model_comparison.csv'):
        results_df = pd.read_csv('reports/model_comparison.csv', index_col=0)
        st.table(results_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image('reports/figures/confusion_matrix.png', caption="Confusion Matrix of Best Model")
    with col2:
        st.image('reports/figures/feature_importance.png', caption="Global Feature Importance")
    
    st.image('reports/figures/correlation_heatmap.png', caption="Feature Correlation Heatmap")

elif page == "Risk Insights":
    st.title("📈 Demographic Disease Trends")
    df = pd.read_csv('data/multi_disease_dataset.csv')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x='Age', hue='Target', fill=True, ax=ax)
    plt.title("Age Distribution by Disease")
    st.pyplot(fig)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Target', y='Glucose', ax=ax2)
    plt.title("Glucose Levels across Classes")
    st.pyplot(fig2)

elif page == "Patient History":
    st.title("📁 Patient History Storage")
    st.info("Feature in development: Database connection for patient records.")
    st.markdown("""
    - [ ] SQL Database Integration
    - [ ] Patient ID Search
    - [ ] Secure Encryption
    """)

# Footer
st.markdown("---")

