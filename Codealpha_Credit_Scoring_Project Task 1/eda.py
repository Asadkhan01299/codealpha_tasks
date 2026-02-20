import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_eda():
    # Load dataset
    df = pd.read_csv('data/german_credit.csv')
    
    # 1. Basic Structure
    print("--- Dataset Info ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    # 2. Target Variable Distribution
    # 1 = Good, 2 = Bad credit risk
    plt.figure(figsize=(6, 4))
    sns.countplot(x='credit_risk', data=df)
    plt.title('Distribution of Credit Risk (1=Good, 2=Bad)')
    plt.savefig('data/risk_distribution.png')
    print("\nSave distribution plot to data/risk_distribution.png")
    
    # 3. Correlation between numerical features
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('data/correlation_heatmap.png')
    print("Save correlation heatmap to data/correlation_heatmap.png")
    
    # 4. Check for missing values
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # 5. Relationship between age and credit risk
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='credit_risk', y='age', data=df)
    plt.title('Age vs Credit Risk')
    plt.savefig('data/age_vs_risk.png')
    print("Save age vs risk plot to data/age_vs_risk.png")

if __name__ == "__main__":
    if not os.path.exists('data/german_credit.csv'):
        print("Dataset not found. Run data_loader.py first.")
    else:
        perform_eda()
