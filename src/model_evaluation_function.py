import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for array creation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder for dummy usage
import os

def evaluate_model(model, X_test, y_test, model_name, le_encoder=None):
    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
    else:
        y_pred_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("") # Newline for separation
    print(f"HASIL EVALUASI {model_name}:")
    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1-Score : {f1:.4f}")

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    xticklabels = le_encoder.classes_ if le_encoder is not None else None
    yticklabels = le_encoder.classes_ if le_encoder is not None else None
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'images/confusion_matrix_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')

    print("") # Newline for separation
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=xticklabels if xticklabels is not None else ['Class 0', 'Class 1']))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Correcting the dummy LabelEncoder usage for string class names
    le_temp = LabelEncoder()
    le_temp.fit(y_train) # This will create classes [0, 1]
    le_temp.classes_ = np.array(['Class 0', 'Class 1']) # Manually set string class names
    model_temp = LogisticRegression()
    model_temp.fit(X_train, y_train)
    results_temp = evaluate_model(model_temp, X_test, y_test, "Test Logistic Regression", le_encoder=le_temp)
    print(results_temp)