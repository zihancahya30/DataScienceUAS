import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_class_distribution(df):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Class', data=df, palette='Set2')
    plt.title('Distribusi Kelas Raisin', fontsize=16, fontweight='bold')
    plt.xlabel('Kelas', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 3,
                f'{height} ({height/len(df)*100:.1f}%)',
                ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('images/class_distribution.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    from load_data import load_raisin_data
    df = load_raisin_data()
    plot_class_distribution(df)
    print("Class distribution plot saved.")