import os

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_default_preprocessing.csv")

    # 1. Ambil path absolut dari folder tempat main.py ini berada
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 2. Cek argumen
    if len(sys.argv) > 3:
        input_path = sys.argv[3]
        # Jika input_path cuma nama file (kayu punya lu sekarang), gabungkan dengan BASE_DIR
        file_path = input_path if os.path.isabs(input_path) else os.path.join(BASE_DIR, input_path)
    else:
        file_path = os.path.join(BASE_DIR, "loan_default_preprocessing.csv")

    try:
        dataset_loan_default_preprocessed = pd.read_csv(file_path)
    except FileNotFoundError:
        # Ambil nama file script ini (main.py)
        current_file = os.path.basename(__file__)
        
        # Langsung tembak isi variabel file_path ke anotasi GitHub
        print(f"::error file={current_file}::Isi variabel file_path saat ini adalah: {file_path}")
        
        # Tetap exit biar workflow-nya sadar kalau ini error
        sys.exit(1)

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