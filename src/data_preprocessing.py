import pandas as pd
import os

def load_and_preprocess_data(data_dir="../data"):
    print("Loading raw data for Version 2...")
    
    # Load required files
    lap_times = pd.read_csv(f"{data_dir}/lap_times.csv")
    races = pd.read_csv(f"{data_dir}/races.csv")
    pit_stops = pd.read_csv(f"{data_dir}/pit_stops.csv")
    results = pd.read_csv(f"{data_dir}/results.csv")
    
    # 1. Merge Races to get Year and Circuit
    df = pd.merge(lap_times, races[['raceId', 'year', 'circuitId']], on='raceId', how='left')
    
    # Filter for V6 Turbo Hybrid era (2014 onwards)
    df = df[df['year'] >= 2014].copy()
    
    # 2. Merge Pit Stops (Binary feature: 1 if pit stop happened on this lap, else 0)
    # We only need raceId, driverId, and lap from pit_stops to match
    pit_stops_simplified = pit_stops[['raceId', 'driverId', 'lap']].copy()
    pit_stops_simplified['is_pit_stop'] = 1
    
    df = pd.merge(df, pit_stops_simplified, on=['raceId', 'driverId', 'lap'], how='left')
    df['is_pit_stop'] = df['is_pit_stop'].fillna(0).astype(int) # Fill non-pit laps with 0
    
    # 3. Merge Results to get Constructor (Car) and Grid Position
    results_simplified = results[['raceId', 'driverId', 'constructorId', 'grid']]
    df = pd.merge(df, results_simplified, on=['raceId', 'driverId'], how='left')
    
    # Select final features and target
    selected_features = ['circuitId', 'driverId', 'lap', 'is_pit_stop', 'constructorId', 'grid']
    target = 'milliseconds'
    
    # Drop any rows with missing values in our selected columns
    df_model = df[selected_features + [target]].dropna()
    
    # Save processed data
    processed_path = f"{data_dir}/processed_f1_data_v2.csv"
    df_model.to_csv(processed_path, index=False)
    print(f"V2 Preprocessing complete. Saved to {processed_path}")
    print(f"Total records for modeling: {len(df_model)}")

if __name__ == "__main__":
    load_and_preprocess_data(data_dir="data")