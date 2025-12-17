import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def prepare_data(df):
    print("Encoding Target Variable...")
    le = LabelEncoder()
    df['Class_encoded'] = le.fit_transform(df['Class'])
    print(f"   Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = df.drop(['Class', 'Class_encoded'], axis=1)
    y = df['Class_encoded']
    print("") # Newline for separation
    print("Split data:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

    print("") # Newline for separation
    print("Split data training dan testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_test shape: {y_test.shape}")

    print("") # Newline for separation
    print("Scaling fitur...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, X, le

if __name__ == '__main__':
    from load_data import load_raisin_data
    df = load_raisin_data()
    X_train_scaled, X_test_scaled, y_train, y_test, X, le = prepare_data(df)
    print("Data preparation complete.")