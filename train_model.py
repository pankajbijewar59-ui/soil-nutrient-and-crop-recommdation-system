"""
train_model.py
---------------
Trains a Random Forest classifier on the soil-nutrient dataset to
predict the most suitable crop, and saves the trained model + scaler
to disk for later use in app.py.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"


def main():
    df = pd.read_csv("dataset.csv")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nModel trained successfully.")
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Feature Importance:")
    print(importances)

    joblib.dump(model, "crop_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nSaved: crop_model.pkl, scaler.pkl")


if __name__ == "__main__":
    main()
