import pandas as pd
import os

def load_and_preprocess_data(data_dir="../data"):
    print("Loading raw data...")
    # Load required files
    lap_times = pd.read_csv(f"{data_dir}/lap_times.csv")
    races = pd.read_csv(f"{data_dir}/races.csv")
    
    # Merge on raceId
    merged_df = pd.merge(lap_times, races[['raceId', 'year', 'circuitId']], on='raceId', how='left')
    
    # Filter for V6 Turbo Hybrid era (2014 onwards)
    df_recent = merged_df[merged_df['year'] >= 2014].copy()
    
    # Select features and target
    selected_features = ['circuitId', 'driverId', 'lap']
    target = 'milliseconds'
    
    # Drop missing values
    df_model = df_recent[selected_features + [target]].dropna()
    
    # Save processed data for the model
    processed_path = f"{data_dir}/processed_f1_data.csv"
    df_model.to_csv(processed_path, index=False)
    print(f"Data preprocessing complete. Saved to {processed_path}")
    print(f"Total records for modeling: {len(df_model)}")

if __name__ == "__main__":
    # Ensure this runs relative to the src folder
    load_and_preprocess_data(data_dir="data")