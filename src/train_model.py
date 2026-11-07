import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_and_evaluate():
    print("Loading processed data...")
    # Load processed data
    try:
        df = pd.read_csv("data/processed_f1_data.csv")
    except FileNotFoundError:
        print("Processed data not found. Run data_preprocessing.py first.")
        return

    X = df[['circuitId', 'driverId', 'lap']]
    y = df['milliseconds']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest model (this might take a minute)...")
    model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    mae_seconds = mean_absolute_error(y_test, preds) / 1000
    print(f"--- Model Evaluation ---")
    print(f"Random Forest MAE: {mae_seconds:.2f} Seconds")

    # Save the model
    import os
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, "models/rf_model_v1.pkl")
    print("Model saved to models/rf_model_v1.pkl")

if __name__ == "__main__":
    train_and_evaluate()
    