import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
from datetime import datetime
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

    n_estimators = 100
    random_state = 42
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    print("🚀 Starting MLflow run...")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"✅ Model trained. Accuracy={acc:.3f}, F1={f1:.3f}")

        # Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Save artifacts locally
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/churn_model.pkl")
        joblib.dump(scaler, "model/scaler.pkl")
        joblib.dump(encoder, "model/encoder.pkl")

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("model/churn_model.pkl")
        mlflow.log_artifact("model/scaler.pkl")
        mlflow.log_artifact("model/encoder.pkl")

        print("📊 Metrics & artifacts logged to MLflow!")

        # ✅ Append metrics to a CSV log
        metrics_path = "model/metrics_history.csv"
        new_record = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": acc,
            "f1_score": f1,
            "n_estimators": n_estimators,
            "random_state": random_state
        }])

        if os.path.exists(metrics_path):
            existing = pd.read_csv(metrics_path)
            updated = pd.concat([existing, new_record], ignore_index=True)
        else:
            updated = new_record

        updated.to_csv(metrics_path, index=False)
        print(f"🧾 Metrics appended to {metrics_path}")


if __name__ == "__main__":
    train_model()
