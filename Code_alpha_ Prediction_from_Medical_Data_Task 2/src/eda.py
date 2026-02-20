import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_name):
    # Determine base name
    base_name = file_name.replace('_processed.csv', '').replace('.csv', '')
    print(f"\n--- EDA for {base_name} ---")
    
    # Load data
    path = os.path.join('data', file_name)
    df = pd.read_csv(path)
    
    # 1. Basic Info
    print("Shape:", df.shape)
    
    # 2. Target Distribution
    target_col = 'target' if 'target' in df.columns else ('Outcome' if 'Outcome' in df.columns else 'diagnosis')
    
    # 3. Correlation Matrix (Numeric only)
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        print(f"Warning: No numeric data found for {base_name}. Skipping correlation plot.")
        return

    plt.figure(figsize=(12, 10))
    # Use a diverging color palette with centering at 0
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', center=0)
    plt.title(f"Correlation Matrix - {base_name}")
    plt.tight_layout()
    
    # Save to models directory
    save_path = f"models/{base_name}_correlation.png"
    plt.savefig(save_path)
    print(f"Correlation plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    if not os.path.exists('models'): os.makedirs('models')
    
    # Prefer processed files if they exist, otherwise use raw
    files_to_process = [
        'heart_disease_processed.csv', 
        'breast_cancer_processed.csv', 
        'diabetes_processed.csv'
    ]
    
    for f in files_to_process:
        if os.path.exists(os.path.join('data', f)):
            perform_eda(f)
        else:
            # Fallback to raw if processed not available
            raw_f = f.replace('_processed', '')
            if os.path.exists(os.path.join('data', raw_f)):
                print(f"Note: Processed file {f} not found, using raw {raw_f}")
                perform_eda(raw_f)
