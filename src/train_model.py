import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_and_evaluate():
    print("Loading V2 processed data...")
    try:
        df = pd.read_csv("data/processed_f1_data_v2.csv")
    except FileNotFoundError:
        print("Error: processed_f1_data_v2.csv not found. Run preprocessing first.")
        return

    # Updated Features
    features = ['circuitId', 'driverId', 'lap', 'is_pit_stop', 'constructorId', 'grid']
    X = df[features]
    y = df['milliseconds']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training V2 Random Forest model (this might take a bit longer)...")
    # Increased n_estimators to 50 for slightly better accuracy
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    mae_seconds = mean_absolute_error(y_test, preds) / 1000
    
    print(f"\n--- Version 2 Model Evaluation ---")
    print(f"Random Forest MAE: {mae_seconds:.2f} Seconds")

    # Save the new model
    joblib.dump(model, "models/rf_model_v2.pkl")
    print("Model saved to models/rf_model_v2.pkl")

if __name__ == "__main__":
    train_and_evaluate()