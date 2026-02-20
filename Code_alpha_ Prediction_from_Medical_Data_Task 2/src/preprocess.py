import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_heart_disease():
    print("Preprocessing Heart Disease data...")
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    # Read without headers, then check if first row is a header
    df = pd.read_csv('data/heart_disease.csv', header=None, names=cols, na_values='?')
    
    # If first row has strings in numeric columns, it's a header
    if not pd.api.types.is_numeric_dtype(df['age']):
        df = df.iloc[1:].reset_index(drop=True)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Target distribution check and filter
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Handle missing values after conversion
    df = df.fillna(df.median())
    
    df.to_csv('data/heart_disease_processed.csv', index=False)
    print(f"Saved heart_disease_processed.csv (Shape: {df.shape})")
    return df

def preprocess_breast_cancer():
    print("Preprocessing Breast Cancer data...")
    cols = ['id', 'diagnosis'] + [f'feat_{i}' for i in range(1, 31)]
    df = pd.read_csv('data/breast_cancer.csv', header=None, names=cols)
    
    # Check for header
    if df.iloc[0]['diagnosis'] == 'diagnosis':
        df = df.iloc[1:].reset_index(drop=True)
    
    # Filter to only 'M' and 'B' (to avoid any other corrupted rows)
    df = df[df['diagnosis'].isin(['M', 'B'])]
    
    # Drop 'id'
    df = df.drop(columns=['id'])
    
    # Encode diagnosis (M=1, B=0)
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    
    # Ensure all features are numeric
    for col in df.columns:
        if col != 'diagnosis':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.fillna(df.median())
    
    df.to_csv('data/breast_cancer_processed.csv', index=False)
    print(f"Saved breast_cancer_processed.csv (Shape: {df.shape})")
    return df

def preprocess_diabetes():
    print("Preprocessing Diabetes data...")
    df = pd.read_csv('data/diabetes.csv')
    if 'Outcome' not in df.columns:
        cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv('data/diabetes.csv', header=None, names=cols)
    
    # Filter invalid outcomes
    if not pd.api.types.is_numeric_dtype(df['Outcome']):
        df['Outcome'] = pd.to_numeric(df['Outcome'], errors='coerce')
        
    df = df.dropna(subset=['Outcome'])
    df['Outcome'] = df['Outcome'].astype(int)
    # Further filter to 0 and 1 only
    df = df[df['Outcome'].isin([0, 1])]
    
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
        
    df.to_csv('data/diabetes_processed.csv', index=False)
    print(f"Saved diabetes_processed.csv (Shape: {df.shape})")
    return df

if __name__ == "__main__":
    if not os.path.exists('data'): os.makedirs('data')
    preprocess_heart_disease()
    preprocess_breast_cancer()
    preprocess_diabetes()
    print("All datasets preprocessed.")
