import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def compare_models(results_lr, results_rf, results_dl):
    comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Deep Learning (MLP)',],
        'Accuracy': [results_lr['accuracy'], results_rf['accuracy'], results_dl['accuracy']],
        'Precision': [results_lr['precision'], results_rf['precision'], results_dl['precision']],
        'Recall': [results_lr['recall'], results_rf['recall'], results_dl['recall']],
        'F1-Score': [results_lr['f1'], results_rf['f1'], results_dl['f1']]
    })

    print("📊 TABEL PERBANDINGAN MODEL:")
    print(comparison_df.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(comparison_df['Model']))
        ax.bar(x, comparison_df[metric], color=colors, alpha=0.8, edgecolor='black')

        ax.set_title(f'Perbandingan {metric}', fontsize=14)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=15)
        ax.set_ylim([0, 1])

        for i, v in enumerate(comparison_df[metric]):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

    plt.suptitle('PERBANDINGAN PERFORMANSI MODEL', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    return comparison_df

if __name__ == '__main__':
    dummy_lr = {'accuracy': 0.88, 'precision': 0.88, 'recall': 0.88, 'f1': 0.88}
    dummy_rf = {'accuracy': 0.85, 'precision': 0.85, 'recall': 0.85, 'f1': 0.85}
    dummy_dl = {'accuracy': 0.87, 'precision': 0.87, 'recall': 0.87, 'f1': 0.87}
    comparison_df = compare_models(dummy_lr, dummy_rf, dummy_dl)
    print("Model comparison complete.")