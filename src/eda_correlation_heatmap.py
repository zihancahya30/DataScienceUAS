import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Heatmap Korelasi Antar Fitur', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    from load_data import load_raisin_data
    df = load_raisin_data()
    plot_correlation_heatmap(df)
    print("Correlation heatmap saved.")