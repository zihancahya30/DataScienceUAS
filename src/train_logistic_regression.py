from sklearn.linear_model import LogisticRegression
from model_evaluation_function import evaluate_model # Relative import

def train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test, le):
    print("") # Newline for separation
    print("TRAINING MODEL BASELINE (Logistic Regression)")
    print("-" * 40)

    model_lr = LogisticRegression(random_state=42, max_iter=1000)
    model_lr.fit(X_train_scaled, y_train)
    results_lr = evaluate_model(model_lr, X_test_scaled, y_test, "Logistic Regression", le_encoder=le)
    return model_lr, results_lr

if __name__ == '__main__':
    # Example usage (assuming necessary data is prepared)
    from load_data import load_raisin_data
    from data_preparation import prepare_data
    df = load_raisin_data()
    X_train_scaled, X_test_scaled, y_train, y_test, X, le = prepare_data(df)
    # Correcting the function call: remove 'X' from the arguments
    model_lr, results_lr = train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test, le)
    print("Logistic Regression training complete.")