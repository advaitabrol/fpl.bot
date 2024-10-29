import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Load the training data
goalkeepers = pd.read_csv('train_data/goalkeeper.csv')
defenders = pd.read_csv('train_data/defender.csv')
midfielders = pd.read_csv('train_data/midfielder.csv')
forwards = pd.read_csv('train_data/forward.csv')

# Fill missing values with 0
goalkeepers.fillna(0, inplace=True)
defenders.fillna(0, inplace=True)
midfielders.fillna(0, inplace=True)
forwards.fillna(0, inplace=True)

# Define the target column
target = 'next_week_points'

# Define the features for each position
goalkeeper_features = [
    'minutes', 'goals_conceded', 'expected_goals_conceded', 'saves', 'penalties_saved', 
    'total_points', 'bonus', 'clean_sheets', 'xA', 'starts', 'form', 
    'clean_sheet_probability', 'last_season_penalties_saved', 'last_season_expected_goals_conceded', 
    'last_season_clean_sheet_probability', 'next_week_specific_fixture_difficulty', 
    'next_week_holistic_fixture_difficulty', 'saves_per_game', 'last_season_total_saves'
]

defender_features = [
    'minutes', 'goals', 'xG', 'assists', 'xA', 'total_points', 'shots', 'key_passes', 'ict_index', 
    'bonus', 'clean_sheets', 'goals_conceded', 'expected_goals_conceded', 'clean_sheet_probability', 
    'last_season_expected_goals_conceded', 'last_season_clean_sheet_probability', 'starts', 
    'form', 'xG&A_form', 'minutes_per_game', 'last_season_xG', 'last_season_xA', 
    'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty'
]

midfielder_features = [
    'goals_conceded', 'clean_sheets', 'minutes', 'goals', 'xG', 'assists', 'xA', 'total_points', 
    'shots', 'key_passes', 'ict_index', 'bonus', 'starts', 'form', 'xG&A_form', 'minutes_per_game', 
    'last_season_goals', 'last_season_assists', 'last_season_xG', 'last_season_xA', 
    'last_season_points_per_minute', 'next_week_specific_fixture_difficulty', 
    'next_week_holistic_fixture_difficulty'
]

forward_features = [
    'minutes', 'goals', 'xG', 'assists', 'xA', 'total_points', 'shots', 'key_passes', 'ict_index', 
    'bonus', 'starts', 'form', 'xG&A_form', 'minutes_per_game', 'last_season_goals', 
    'last_season_assists', 'last_season_xG', 'last_season_xA', 'last_season_points_per_minute', 
    'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty'
]

# Split data into training and test sets for each position
X_gk, y_gk = goalkeepers[goalkeeper_features], goalkeepers[target]
X_train_gk, X_test_gk, y_train_gk, y_test_gk = train_test_split(X_gk, y_gk, test_size=0.2, random_state=42)

X_def, y_def = defenders[defender_features], defenders[target]
X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(X_def, y_def, test_size=0.2, random_state=42)

X_mid, y_mid = midfielders[midfielder_features], midfielders[target]
X_train_mid, X_test_mid, y_train_mid, y_test_mid = train_test_split(X_mid, y_mid, test_size=0.2, random_state=42)

X_fwd, y_fwd = forwards[forward_features], forwards[target]
X_train_fwd, X_test_fwd, y_train_fwd, y_test_fwd = train_test_split(X_fwd, y_fwd, test_size=0.2, random_state=42)

# Train the Random Forest models for each position
def train_rf_model(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Train models for each position
rf_gk_model = train_rf_model(X_train_gk, y_train_gk)
rf_def_model = train_rf_model(X_train_def, y_train_def)
rf_mid_model = train_rf_model(X_train_mid, y_train_mid)
rf_fwd_model = train_rf_model(X_train_fwd, y_train_fwd)

# Function to retain only specific columns in a CSV
def retain_columns(csv_file):
    """
    Retain only the columns 'name', 'team', 'price', and 'predicted_next_week_points' in the CSV file.
    """
    df = pd.read_csv(csv_file)

    # Check if all the required columns are present
    required_columns = ['name', 'team', 'price', 'predicted_next_week_points']
    available_columns = [col for col in required_columns if col in df.columns]
    
    if len(available_columns) == len(required_columns):
        # Retain only the specified columns
        df = df[required_columns]
        df.to_csv(csv_file, index=False)
        print(f"Updated {csv_file} with only the required columns.")
    else:
        missing_cols = set(required_columns) - set(available_columns)
        print(f"Missing columns in {csv_file}: {missing_cols}")

# Apply trained model to prediction data and save results
def apply_trained_model(model, features, csv_file):
    """
    Apply a trained model to a CSV file, make predictions, and save the results.
    """
    df = pd.read_csv(csv_file)

    if all(feature in df.columns for feature in features):
        X = df[features]
        df['predicted_next_week_points'] = model.predict(X)
        df.to_csv(csv_file, index=False)
        print(f"Predictions saved to {csv_file}")
        
        # Now retain only specific columns
        retain_columns(csv_file)
    else:
        missing_features = [f for f in features if f not in df.columns]
        print(f"Missing features in {csv_file}: {missing_features}")

# Apply models to prediction data in prediction_data/2024-25 folder
apply_trained_model(rf_gk_model, goalkeeper_features, 'prediction_data/2024-25/GW9/gk.csv')
apply_trained_model(rf_def_model, defender_features, 'prediction_data/2024-25/GW9/def.csv')
apply_trained_model(rf_mid_model, midfielder_features, 'prediction_data/2024-25/GW9/mid.csv')
apply_trained_model(rf_fwd_model, forward_features, 'prediction_data/2024-25/GW9/fwd.csv')

# Evaluate model performance on the test sets
def evaluate_model(model, X_test, y_test, label):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"{label} - Random Forest MSE: {mse}")
    print(f"{label} - Random Forest MAE: {mae}")

# Evaluate all models
evaluate_model(rf_gk_model, X_test_gk, y_test_gk, "Goalkeepers")
evaluate_model(rf_def_model, X_test_def, y_test_def, "Defenders")
evaluate_model(rf_mid_model, X_test_mid, y_test_mid, "Midfielders")
evaluate_model(rf_fwd_model, X_test_fwd, y_test_fwd, "Forwards")