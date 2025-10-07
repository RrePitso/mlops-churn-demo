import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_model(data_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    print("📦 Loading data...")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.replace(' ', '_')
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    cat_cols = df.select_dtypes('object').columns
    encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Model trained. Accuracy={acc:.3f}, F1={f1:.3f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/churn_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(encoder, "model/encoder.pkl")

    print("💾 Model, scaler, and encoder saved successfully!")

if __name__ == "__main__":
    train_model()
