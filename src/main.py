import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


from load_data import load_raisin_data
from data_preparation import prepare_data
from eda_class_distribution import plot_class_distribution
from eda_correlation_heatmap import plot_correlation_heatmap
from eda_numerical_feature_distribution import plot_numerical_feature_distribution
from train_logistic_regression import train_logistic_regression
from train_random_forest import train_random_forest
from train_deep_learning_mlp import train_deep_learning_mlp
from model_comparison import compare_models
from save_models import save_all_models

def main():
    print("Starting the Raisin Classification Workflow...")
    print("") # Added for newline

    print("Step 1: Loading data...")
    df = load_raisin_data()
    print("Data loaded.")
    print("") # Newline for separation

    print("Step 2: Preparing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, X, le = prepare_data(df)
    print("Data preparation complete.")
    print("") # Newline for separation

    print("Step 3: Performing EDA and saving plots...")
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_numerical_feature_distribution(df)
    print("EDA plots generated and saved in 'images' directory.")
    print("") # Newline for separation

    print("Step 4: Training and evaluating models...")
    print("") # Newline for separation

    model_lr, results_lr = train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test, le)
    print("Logistic Regression training and evaluation complete.")
    print("") # Newline for separation

    model_rf, results_rf = train_random_forest(X_train_scaled, X_test_scaled, y_train, y_test, X, le)
    print("Random Forest training and evaluation complete.")
    print("") # Newline for separation

    model_mlp, results_dl = train_deep_learning_mlp(X_train_scaled, X_test_scaled, y_train, y_test, le)
    print("Deep Learning (MLP) training and evaluation complete.")
    print("") # Newline for separation

    print("Step 5: Comparing models...")
    comparison_df = compare_models(results_lr, results_rf, results_dl)
    print("Model comparison complete and plot saved in 'images' directory.")
    print("") # Newline for separation

    print("Step 6: Saving models...")
    save_all_models(model_lr, model_rf, model_mlp, comparison_df)
    print("Models saved in 'models' directory.")
    print("") # Newline for separation

    print("Raisin Classification Workflow Finished Successfully!")

if __name__ == '__main__':
    main()