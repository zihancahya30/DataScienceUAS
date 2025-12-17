from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from model_evaluation_function import evaluate_model # Relative import

def train_random_forest(X_train_scaled, X_test_scaled, y_train, y_test, X_features, le):
    print("") # Newline for separation
    print("TRAINING MODEL ADVANCED (Random Forest)")
    print("-" * 40)

    model_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model_rf.fit(X_train_scaled, y_train)
    results_rf = evaluate_model(model_rf, X_test_scaled, y_test, "Random Forest", le_encoder=le)

    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': X_features.columns,
        'importance': model_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
    plt.title('Feature Importance - Random Forest', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
    return model_rf, results_rf

if __name__ == '__main__':
    from load_data import load_raisin_data
    from data_preparation import prepare_data
    df = load_raisin_data()
    X_train_scaled, X_test_scaled, y_train, y_test, X, le = prepare_data(df)
    model_rf, results_rf = train_random_forest(X_train_scaled, X_test_scaled, y_train, y_test, X, le)
    print("Random Forest training complete.")