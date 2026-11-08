# F1 Lap Time Predictor 🏎️

A Machine Learning project to predict Formula 1 lap times using telemetry and historical race data. 

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-Learn (Random Forest Regressor)

## Setup Instructions
1. Clone the repo: `git clone [your-repo-link]`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the Kaggle F1 dataset and place the CSV files in the `data/` folder.
4. Run preprocessing: `python src/data_preprocessing.py`
5. Train model: `python src/train_model.py`

## Current Performance
- Baseline Random Forest MAE: ~9 seconds