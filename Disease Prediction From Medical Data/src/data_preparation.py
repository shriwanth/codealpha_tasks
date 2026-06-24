import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=500):
    np.random.seed(42)
    
    # Define features
    features = [
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
        'RestECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'Pregnancies', 'Glucose', 
        'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
        'Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean', 
        'Smoothness_mean', 'Compactness_mean', 'Concavity_mean'
    ]
    
    data_list = []
    
    # 1. Heart Disease Samples
    for _ in range(n_samples):
        row = {f: 0.0 for f in features}
        row['Age'] = np.random.randint(40, 75)
        row['Sex'] = np.random.choice([0, 1])
        row['ChestPainType'] = np.random.randint(1, 4)
        row['RestingBP'] = np.random.randint(130, 180)
        row['Cholesterol'] = np.random.randint(200, 350)
        row['FastingBS'] = np.random.choice([0, 1], p=[0.3, 0.7])
        row['RestECG'] = np.random.randint(0, 2)
        row['MaxHR'] = np.random.randint(100, 150)
        row['ExerciseAngina'] = np.random.choice([0, 1], p=[0.2, 0.8])
        row['Oldpeak'] = np.random.uniform(1.5, 4.0)
        # Other features normal
        row['Glucose'] = np.random.randint(70, 120)
        row['BMI'] = np.random.uniform(18, 30)
        row['BloodPressure'] = row['RestingBP']
        row['Target'] = 'Heart Disease'
        data_list.append(row)

    # 2. Diabetes Samples
    for _ in range(n_samples):
        row = {f: 0.0 for f in features}
        row['Age'] = np.random.randint(20, 70)
        row['Pregnancies'] = np.random.randint(0, 10)
        row['Glucose'] = np.random.randint(140, 200)
        row['BloodPressure'] = np.random.randint(80, 110)
        row['SkinThickness'] = np.random.randint(20, 50)
        row['Insulin'] = np.random.randint(100, 300)
        row['BMI'] = np.random.uniform(28, 45)
        row['DiabetesPedigreeFunction'] = np.random.uniform(0.5, 2.0)
        row['RestingBP'] = row['BloodPressure']
        row['Target'] = 'Diabetes'
        data_list.append(row)

    # 3. Breast Cancer Samples
    for _ in range(n_samples):
        row = {f: 0.0 for f in features}
        row['Age'] = np.random.randint(30, 80)
        row['Radius_mean'] = np.random.uniform(15, 25)
        row['Texture_mean'] = np.random.uniform(20, 35)
        row['Perimeter_mean'] = np.random.uniform(100, 170)
        row['Area_mean'] = np.random.uniform(700, 2000)
        row['Smoothness_mean'] = np.random.uniform(0.1, 0.15)
        row['Compactness_mean'] = np.random.uniform(0.15, 0.3)
        row['Concavity_mean'] = np.random.uniform(0.15, 0.4)
        row['Target'] = 'Breast Cancer'
        data_list.append(row)

    # 4. No Disease Samples
    for _ in range(n_samples):
        row = {f: 0.0 for f in features}
        row['Age'] = np.random.randint(18, 60)
        row['Sex'] = np.random.choice([0, 1])
        row['ChestPainType'] = 0
        row['RestingBP'] = np.random.randint(110, 130)
        row['Cholesterol'] = np.random.randint(150, 200)
        row['FastingBS'] = 0
        row['RestECG'] = 0
        row['MaxHR'] = np.random.randint(150, 190)
        row['ExerciseAngina'] = 0
        row['Oldpeak'] = 0.0
        row['Pregnancies'] = np.random.randint(0, 3)
        row['Glucose'] = np.random.randint(70, 100)
        row['BloodPressure'] = np.random.randint(70, 85)
        row['BMI'] = np.random.uniform(18.5, 25)
        row['Radius_mean'] = np.random.uniform(10, 14)
        row['Target'] = 'No Disease'
        data_list.append(row)

    df = pd.DataFrame(data_list)
    
    # Add some noise to all features to make it more realistic
    for col in features:
        if df[col].dtype == 'float64':
             df[col] += np.random.normal(0, df[col].std() * 0.05, size=len(df))
    
    # Ensure no negative values for biological metrics
    for col in features:
        if col not in ['Oldpeak']: # Oldpeak can be 0, but usually positive
            df[col] = df[col].apply(lambda x: max(0, x))

    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/multi_disease_dataset.csv', index=False)
    print(f"Dataset created with {len(df)} samples and saved to data/multi_disease_dataset.csv")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
