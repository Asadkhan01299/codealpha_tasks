import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import json

def train_on_dataset(file_path):
    print(f"\nTraining on {file_path}...")
    df = pd.read_csv(file_path)
    
    # Identify target column
    target_col = 'target' if 'target' in df.columns else ('Outcome' if 'Outcome' in df.columns else 'diagnosis')
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  Running {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': acc,
            'f1_score': f1
        }
        print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
    return results

if __name__ == "__main__":
    all_results = {}
    datasets = [
        'data/heart_disease_processed.csv',
        'data/breast_cancer_processed.csv',
        'data/diabetes_processed.csv'
    ]
    
    for ds in datasets:
        if os.path.exists(ds):
            name = ds.split('/')[-1].replace('_processed.csv', '')
            all_results[name] = train_on_dataset(ds)
            
    # Save results to JSON
    with open('models/results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\nResults saved to models/results.json")
