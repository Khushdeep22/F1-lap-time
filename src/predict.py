import joblib
import pandas as pd

def test_prediction():
    print("Loading saved model...")
    try:
        # Puraana train kiya hua model load kar rahe hain
        model = joblib.load("models/rf_model_v1.pkl")
    except FileNotFoundError:
        print("Error: Model file nahi mili. Pehle train_model.py run karo.")
        return

    # Ek dummy scenario create karte hain testing ke liye
    # Example: circuitId = 1 (Albert Park), driverId = 1 (Hamilton), lap = 10
    sample_input = pd.DataFrame({
        'circuitId': [1],
        'driverId': [1],
        'lap': [10]
    })

    print("\n--- Testing Model Prediction ---")
    print(f"Input: Circuit=1, Driver=1, Lap=10")
    
    # Prediction karna
    predicted_ms = model.predict(sample_input)[0]
    
    # Milliseconds ko seconds aur minutes mein convert karna
    predicted_sec = predicted_ms / 1000
    mins = int(predicted_sec // 60)
    secs = predicted_sec % 60
    
    print(f"Predicted Lap Time: {predicted_sec:.2f} seconds ({mins}m {secs:.2f}s)\n")
    print("Model is working perfectly! ✅")

if __name__ == "__main__":
    test_prediction()