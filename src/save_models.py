import os
import joblib
from tensorflow.keras.models import save_model

def save_all_models(model_lr, model_rf, model_mlp, comparison_df):
    best_model_idx = comparison_df['Accuracy'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    best_accuracy = comparison_df.loc[best_model_idx, 'Accuracy']

    print("") # Newline for separation
    print(f"MODEL TERBAIK: {best_model_name}")
    print(f"   Dengan akurasi: {best_accuracy:.4f}")

    joblib.dump(model_lr, 'models/logistic_regression_model.pkl')
    print("Logistic Regression model saved to models/logistic_regression_model.pkl")

    joblib.dump(model_rf, 'models/random_forest_model.pkl')
    print("Random Forest model saved to models/random_forest_model.pkl")

    model_mlp.save('models/deep_learning_mlp_model.h5')
    print("Deep Learning (MLP) model saved to models/deep_learning_mlp_model.h5")

if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from tensorflow import keras
    import pandas as pd
    import numpy as np

    class DummyMLP(keras.Model):
        def __init__(self):
            super().__init__()
            self.dense = keras.layers.Dense(1, activation='sigmoid')
        def call(self, inputs):
            return self.dense(inputs)
    dummy_model_mlp = DummyMLP()
    dummy_model_mlp.build(input_shape=(None, 10))

    dummy_model_lr = LogisticRegression()
    dummy_model_rf = RandomForestClassifier()
    dummy_comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Deep Learning (MLP)'],
        'Accuracy': [0.9, 0.85, 0.88],
        'Precision': [0.9, 0.85, 0.88],
        'Recall': [0.9, 0.85, 0.88],
        'F1-Score': [0.9, 0.85, 0.88]
    })
    save_all_models(dummy_model_lr, dummy_model_rf, dummy_model_mlp, dummy_comparison_df)
    print("Dummy models saved.")