import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "telecom_sites.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    
    features = ["dg_hours", "battery_voltage", "temperature", "fuel_level", "rectifier_alarm"]
    target = "failure"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Model Training Complete. Test Accuracy: {accuracy_score(y_test, preds):.2f}")
    
    model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, "xgb_telecom.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train()
