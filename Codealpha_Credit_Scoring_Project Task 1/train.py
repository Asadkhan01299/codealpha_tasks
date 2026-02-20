import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os

def train_models():
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_roc_auc = 0
    best_name = ""
    
    print("--- Training and Initial Evaluation ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nModel: {name}")
        print(f"ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        
        if auc > best_roc_auc:
            best_roc_auc = auc
            best_model = model
            best_name = name
            
    print(f"\nBest Model: {best_name} with ROC-AUC {best_roc_auc:.4f}")
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/credit_model.pkl')
    print(f"Saved best model to models/credit_model.pkl")

if __name__ == "__main__":
    train_models()
