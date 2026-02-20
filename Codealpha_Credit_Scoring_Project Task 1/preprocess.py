import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_data():
    df = pd.read_csv('data/german_credit.csv')
    
    # 1. Target Mapping: 1 (Good) -> 1, 2 (Bad) -> 0
    df['credit_risk'] = df['credit_risk'].map({1: 1, 2: 0})
    
    # 2. Identify Categorical and Numerical Columns
    # Based on eda.py and dataset description, most are categorical strings like 'A11'
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target from numerical columns
    if 'credit_risk' in numerical_cols:
        numerical_cols.remove('credit_risk')
    
    # 3. One-Hot Encoding for categorical features
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 4. Feature Scaling for numerical features
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    # 5. Split Data
    X = df_processed.drop('credit_risk', axis=1)
    y = df_processed['credit_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save processed data and scaler
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Preprocessing complete.")
    print(f"Features after encoding: {X_train.shape[1]}")
    print("Files saved to data/processed/")

if __name__ == "__main__":
    preprocess_data()
