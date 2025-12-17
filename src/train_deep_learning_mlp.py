import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

def train_deep_learning_mlp(X_train_scaled, X_test_scaled, y_train, y_test, le):
    print("") # Newline for separation
    print("TRAINING DEEP LEARNING MODEL (MLP)")
    print("-" * 40)

    model_mlp = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model_mlp.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    print("") # Newline for separation
    print("START TRAINING...")
    history = model_mlp.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    print("Training finished.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Training & Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Deep Learning Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/history_loss_accuracy.png', dpi=300, bbox_inches='tight')

    print("") # Newline for separation
    print("EVALUASI MODEL DEEP LEARNING:")
    loss, accuracy = model_mlp.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"   Test Loss: {loss:.4f}")
    print(f"   Test Accuracy: {accuracy:.4f}")

    y_pred_dl = (model_mlp.predict(X_test_scaled, verbose=0) > 0.5).astype(int).flatten()

    accuracy_dl = accuracy_score(y_test, y_pred_dl)
    precision_dl = precision_score(y_test, y_pred_dl, average='weighted')
    recall_dl = recall_score(y_test, y_pred_dl, average='weighted')
    f1_dl = f1_score(y_test, y_pred_dl, average='weighted')

    results_dl = {
        'accuracy': accuracy_dl,
        'precision': precision_dl,
        'recall': recall_dl,
        'f1': f1_dl,
        'y_pred': y_pred_dl
    }

    plt.figure(figsize=(6, 5))
    cm_dl = confusion_matrix(y_test, y_pred_dl)
    sns.heatmap(cm_dl, annot=True, fmt='d', cmap='Greens',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Deep Learning (MLP)', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('images/confusion_matrix_deep_learning.png', dpi=300, bbox_inches='tight')

    print("") # Newline for separation
    print("CLASSIFICATION REPORT - Deep Learning:")
    print(classification_report(y_test, y_pred_dl, target_names=le.classes_))
    return model_mlp, results_dl

if __name__ == '__main__':
    from load_data import load_raisin_data
    from data_preparation import prepare_data
    df = load_raisin_data()
    X_train_scaled, X_test_scaled, y_train, y_test, X, le = prepare_data(df)
    model_mlp, results_dl = train_deep_learning_mlp(X_train_scaled, X_test_scaled, y_train, y_test, le)
    print("Deep Learning MLP training complete.")