import os

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
import sys
from pathlib import Path

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    
    # Ambil lokasi folder tempat main.py berada
    SCRIPT_DIR = Path(__file__).resolve().parent

    # Ambil input dari argumen atau pakai default nama file
    input_arg = sys.argv[3] if len(sys.argv) > 3 else "loan_default_preprocessing.csv"

    # Jika argumennya cuma nama file, otomatis digabung dengan folder SCRIPT_DIR
    file_path = Path(input_arg) if Path(input_arg).is_absolute() else SCRIPT_DIR / input_arg

    dataset_loan_default_preprocessed = pd.read_csv(file_path)

    X = dataset_loan_default_preprocessed.drop("Default", axis=1)
    y = dataset_loan_default_preprocessed["Default"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # log parameters
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 136
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 23

    with mlflow.start_run() as run:
        model_loan_default = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42
        )

        model_loan_default.fit(X_train, y_train)

        predicted_qualities = model_loan_default.predict(X_test)

        mlflow.sklearn.log_model(
            sk_model=model_loan_default,
            artifact_path="model_loan_default"
        )

        # Log metrics
        accuracy = model_loan_default.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)