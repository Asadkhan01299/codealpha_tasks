import pandas as pd
import os

def download_data():
    # URL for the German Credit Dataset (UCI mirror)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    # Columns as per the dataset documentation (german.doc)
    columns = [
        'checking_account', 'duration', 'credit_history', 'purpose', 'amount',
        'savings_account', 'employment_since', 'installment_rate', 'personal_status',
        'other_debtors', 'residence_since', 'property', 'age', 'installment_plans',
        'housing', 'number_credits', 'job', 'people_liable', 'telephone',
        'foreign_worker', 'credit_risk'
    ]
    
    print("Downloading German Credit Dataset...")
    try:
        import requests
        import io
        
        # Use requests with verify=False to bypass SSL issues if necessary
        response = requests.get(url, verify=False)
        response.raise_for_status()
        
        # The data is space-separated
        df = pd.read_csv(io.StringIO(response.text), sep=' ', header=None, names=columns)
        
        # Save to local data folder
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/german_credit.csv', index=False)
        print("Success: Dataset saved to data/german_credit.csv")
        return df
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_data()
