import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_numerical_feature_distribution(df):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    for i, feature in enumerate(numerical_features):
        if i < len(axes):
            ax = axes[i]
            sns.histplot(df[feature], kde=True, ax=ax, color='skyblue')
            ax.set_title(f'Distribusi {feature}', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('')

    for i in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle('Distribusi Fitur Numerik', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/numerical_feature_distribution.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    from load_data import load_raisin_data
    df = load_raisin_data()
    plot_numerical_feature_distribution(df)
    print("Numerical feature distribution plot saved.")