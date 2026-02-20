import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import os

def evaluate_model():
    # Load data
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Load model
    model = joblib.load('models/credit_model.pkl')
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 1. Classification Report
    print("--- Final Classification Report ---")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # 2. Confusion Matrix
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad', 'Good'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('reports/confusion_matrix.png')
    print("Saved Confusion Matrix to reports/confusion_matrix.png")
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('reports/roc_curve.png')
    print("Saved ROC Curve to reports/roc_curve.png")
    
    # 4. Save metrics to a text file
    with open('reports/evaluation_metrics.txt', 'w') as f:
        f.write("--- Evaluation Metrics ---\n")
        f.write(report)
        f.write(f"\nROC-AUC Score: {roc_auc:.4f}\n")

if __name__ == "__main__":
    evaluate_model()
