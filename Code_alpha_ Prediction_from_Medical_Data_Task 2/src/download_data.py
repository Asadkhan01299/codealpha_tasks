import os
import pandas as pd
import requests

def download_datasets():
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    datasets = {
        'heart_disease': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
            'columns': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        },
        'breast_cancer': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
            'columns': ['id', 'diagnosis'] + [f'feat_{i}' for i in range(1, 31)]
        },
        'diabetes': {
            'url': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
            'columns': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        }
    }

    for name, info in datasets.items():
        print(f"Downloading {name} from {info['url']}...")
        try:
            # Read CSV data directly from URL
            # Note: Cleveland heart disease uses '?' for missing values
            df = pd.read_csv(info['url'], names=info['columns'], na_values='?')
            
            output_path = os.path.join('data', f'{name}.csv')
            df.to_csv(output_path, index=False)
            print(f"Successfully saved {name} to {output_path} (Shape: {df.shape})")
        except Exception as e:
            print(f"Error downloading {name}: {e}")

if __name__ == "__main__":
    download_datasets()
