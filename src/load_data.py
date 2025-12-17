import pandas as pd
import numpy as np
import warnings

def load_raisin_data(filepath='/content/Raisin_Dataset.xlsx'):
    warnings.filterwarnings('ignore')
    df = pd.read_excel(filepath)
    print(f"Shape: {df.shape}")
    return df

if __name__ == '__main__':
    df = load_raisin_data()
    print("Data loaded successfully.")