import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import os
from fuzzywuzzy import process

# Load the training data
goalkeepers = pd.read_csv('train_data/goalkeeper.csv')
defenders = pd.read_csv('train_data/defender.csv')
midfielders = pd.read_csv('train_data/midfielder.csv')
forwards = pd.read_csv('train_data/forward.csv')

target='next_week_points'

# Advanced feature engineering

# Goal and Assist Stats for Midfielders and Forwards
midfielders['goal_stat'] = midfielders['last_season_goals'] + midfielders['last_season_xG']
midfielders['assist_stat'] = midfielders['last_season_assists'] + midfielders['last_season_xA']
forwards['goal_stat'] = forwards['last_season_goals'] + forwards['last_season_xG']
forwards['assist_stat'] = forwards['last_season_assists'] + forwards['last_season_xA']

# Moving Averages for Recent Form - 3 and 5 game averages, now including 'bonus'
for df, features in [(midfielders, ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus', 'key_passes', 'shots']),
                     (forwards, ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus', 'shots', 'key_passes']),
                     (defenders, ['clean_sheets', 'goals', 'xG', 'goals_conceded', 'expected_goals_conceded', 'ict_index', 'bonus']),
                     (goalkeepers, ['clean_sheets', 'saves', 'goals_conceded', 'expected_goals_conceded', 'bonus'])]:
    for feature in features:
        df[f'{feature}_3g_avg'] = df.groupby('name')[feature].transform(lambda x: x.fillna(0).rolling(3, min_periods=1).mean())
        df[f'{feature}_5g_avg'] = df.groupby('name')[feature].transform(lambda x: x.fillna(0).rolling(5, min_periods=1).mean())

# Per-90 Stats - Calculate Cumulative Per-90 Stats
for df, feature_list in [(defenders, ['clean_sheets', 'goals', 'xG', 'assists', 'ict_index']), 
                         (midfielders, ['goals', 'assists', 'key_passes', 'shots', 'ict_index']),
                         (forwards, ['goals', 'assists', 'ict_index']),
                         (goalkeepers, ['saves'])]:
    # Calculate cumulative minutes for each player up to each game
    df['cumulative_minutes'] = df.groupby('name')['minutes'].cumsum()
    
    for feature in feature_list:
        # Calculate cumulative total for each feature up to each game
        df[f'cumulative_{feature}'] = df.groupby('name')[feature].cumsum()
        
        # Calculate per-90 stats based on cumulative values up to each game
        df[f'{feature}_per_90'] = np.where(df['cumulative_minutes'] > 0, 
                                           df[f'cumulative_{feature}'] / (df['cumulative_minutes'] / 90), 0)
# Drop helper columns after calculating per-90 stats
for df in [goalkeepers, defenders, midfielders, forwards]:
    feature_list = ['clean_sheets', 'goals', 'xG', 'assists', 'key_passes', 'shots', 'saves', 'ict_index']
    df.drop(columns=['cumulative_minutes'] + [f'cumulative_{feature}' for feature in feature_list if f'cumulative_{feature}' in df.columns], inplace=True)

    # Calculate 3-game and 5-game rolling averages for fixture difficulties
for df in [goalkeepers, defenders, midfielders, forwards]:
    # Check if the columns exist before performing the calculation
    if 'next_week_specific_fixture_difficulty' in df.columns:
        df['next_week_specific_fixture_difficulty_3g_avg'] = df.groupby('name')['next_week_specific_fixture_difficulty'].transform(lambda x: x.fillna(0).rolling(3, min_periods=1).mean())
        df['next_week_specific_fixture_difficulty_5g_avg'] = df.groupby('name')['next_week_specific_fixture_difficulty'].transform(lambda x: x.fillna(0).rolling(5, min_periods=1).mean())

    if 'next_week_holistic_fixture_difficulty' in df.columns:
        df['next_week_holistic_fixture_difficulty_3g_avg'] = df.groupby('name')['next_week_holistic_fixture_difficulty'].transform(lambda x: x.fillna(0).rolling(3, min_periods=1).mean())
        df['next_week_holistic_fixture_difficulty_5g_avg'] = df.groupby('name')['next_week_holistic_fixture_difficulty'].transform(lambda x: x.fillna(0).rolling(5, min_periods=1).mean())


# Interaction Terms based on 3-game and 5-game rolling averages
# Defenders
defenders['clean_sheets_goals_3g'] = defenders['clean_sheets_3g_avg'] * defenders['goals_3g_avg']
defenders['ict_clean_sheets_3g'] = defenders['ict_index_3g_avg'] * defenders['clean_sheets_3g_avg']
defenders['xG_exp_goals_conceded_3g'] = defenders['xG_3g_avg'] * defenders['expected_goals_conceded_3g_avg']

defenders['clean_sheets_goals_5g'] = defenders['clean_sheets_5g_avg'] * defenders['goals_5g_avg']
defenders['ict_clean_sheets_5g'] = defenders['ict_index_5g_avg'] * defenders['clean_sheets_5g_avg']
defenders['xG_exp_goals_conceded_5g'] = defenders['xG_5g_avg'] * defenders['expected_goals_conceded_5g_avg']

# Midfielders
midfielders['key_passes_assists_3g'] = midfielders['key_passes_3g_avg'] * midfielders['assists_3g_avg']
midfielders['shots_goals_3g'] = midfielders['shots_3g_avg'] * midfielders['goals_3g_avg']

midfielders['key_passes_assists_5g'] = midfielders['key_passes_5g_avg'] * midfielders['assists_5g_avg']
midfielders['shots_goals_5g'] = midfielders['shots_5g_avg'] * midfielders['goals_5g_avg']

# Forwards
forwards['goal_ict_index_3g'] = forwards['goal_stat'] * forwards['ict_index_3g_avg']
forwards['goal_ict_index_5g'] = forwards['goal_stat'] * forwards['ict_index_5g_avg']# Goal and Assist Stats for Midfielders and Forwards
midfielders['goal_stat'] = midfielders['last_season_goals'] + midfielders['last_season_xG']
midfielders['assist_stat'] = midfielders['last_season_assists'] + midfielders['last_season_xA']
forwards['goal_stat'] = forwards['last_season_goals'] + forwards['last_season_xG']
forwards['assist_stat'] = forwards['last_season_assists'] + forwards['last_season_xA']

# Moving Averages for Recent Form - 3 and 5 game averages, now including 'bonus'
for df, features in [(midfielders, ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus']),
                     (forwards, ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus']),
                     (defenders, ['clean_sheets', 'goals', 'xG', 'goals_conceded', 'expected_goals_conceded', 'ict_index', 'bonus']),
                     (goalkeepers, ['clean_sheets', 'saves', 'goals_conceded', 'expected_goals_conceded', 'bonus'])]:
    for feature in features:
        if feature in df.columns:  # Only calculate if the feature is present
            df[f'{feature}_3g_avg'] = df.groupby('name')[feature].transform(lambda x: x.fillna(0).rolling(3, min_periods=1).mean())
            df[f'{feature}_5g_avg'] = df.groupby('name')[feature].transform(lambda x: x.fillna(0).rolling(5, min_periods=1).mean())

# Per-90 Stats Calculation with cumulative stats
for df, feature_list in [(defenders, ['clean_sheets', 'goals', 'xG', 'assists', 'ict_index']), 
                         (midfielders, ['goals', 'assists', 'key_passes', 'shots', 'ict_index']),
                         (forwards, ['goals', 'assists', 'ict_index']),
                         (goalkeepers, ['saves'])]:
    # Calculate cumulative minutes for each player up to each game
    df['cumulative_minutes'] = df.groupby('name')['minutes'].cumsum()
    
    for feature in feature_list:
        if feature in df.columns:
            # Calculate cumulative total for each feature up to each game
            df[f'cumulative_{feature}'] = df.groupby('name')[feature].cumsum()
            # Calculate per-90 stats based on cumulative values up to each game
            df[f'{feature}_per_90'] = np.where(df['cumulative_minutes'] > 0, 
                                               df[f'cumulative_{feature}'] / (df['cumulative_minutes'] / 90), 0)

# Drop helper columns after calculating per-90 stats
for df in [goalkeepers, defenders, midfielders, forwards]:
    feature_list = ['clean_sheets', 'goals', 'xG', 'assists', 'key_passes', 'shots', 'saves', 'ict_index']
    df.drop(columns=['cumulative_minutes'] + [f'cumulative_{feature}' for feature in feature_list if f'cumulative_{feature}' in df.columns], inplace=True)

# Interaction Terms based on 3-game and 5-game rolling averages
# Defenders
if 'clean_sheets_3g_avg' in defenders.columns and 'goals_3g_avg' in defenders.columns:
    defenders['clean_sheets_goals_3g'] = defenders['clean_sheets_3g_avg'] * defenders['goals_3g_avg']
if 'ict_index_3g_avg' in defenders.columns and 'clean_sheets_3g_avg' in defenders.columns:
    defenders['ict_index_clean_sheets_3g'] = defenders['ict_index_3g_avg'] * defenders['clean_sheets_3g_avg']
if 'xG_3g_avg' in defenders.columns and 'expected_goals_conceded_3g_avg' in defenders.columns:
    defenders['xG_expected_goals_conceded_3g'] = defenders['xG_3g_avg'] * defenders['expected_goals_conceded_3g_avg']

if 'clean_sheets_5g_avg' in defenders.columns and 'goals_5g_avg' in defenders.columns:
    defenders['clean_sheets_goals_5g'] = defenders['clean_sheets_5g_avg'] * defenders['goals_5g_avg']
if 'ict_index_5g_avg' in defenders.columns and 'clean_sheets_5g_avg' in defenders.columns:
    defenders['ict_index_clean_sheets_5g'] = defenders['ict_index_5g_avg'] * defenders['clean_sheets_5g_avg']
if 'xG_5g_avg' in defenders.columns and 'expected_goals_conceded_5g_avg' in defenders.columns:
    defenders['xG_expected_goals_conceded_5g'] = defenders['xG_5g_avg'] * defenders['expected_goals_conceded_5g_avg']

# Midfielders
if 'key_passes_3g_avg' in midfielders.columns and 'assists_3g_avg' in midfielders.columns:
    midfielders['key_passes_assists_3g'] = midfielders['key_passes_3g_avg'] * midfielders['assists_3g_avg']
if 'shots_3g_avg' in midfielders.columns and 'goals_3g_avg' in midfielders.columns:
    midfielders['shots_goals_3g'] = midfielders['shots_3g_avg'] * midfielders['goals_3g_avg']

if 'key_passes_5g_avg' in midfielders.columns and 'assists_5g_avg' in midfielders.columns:
    midfielders['key_passes_assists_5g'] = midfielders['key_passes_5g_avg'] * midfielders['assists_5g_avg']
if 'shots_5g_avg' in midfielders.columns and 'goals_5g_avg' in midfielders.columns:
    midfielders['shots_goals_5g'] = midfielders['shots_5g_avg'] * midfielders['goals_5g_avg']

# Forwards
if 'goals_3g_avg' in forwards.columns and 'ict_index_3g_avg' in forwards.columns:
    forwards['goals_ict_index_3g'] = forwards['goals_3g_avg'] * forwards['ict_index_3g_avg']
if 'goals_5g_avg' in forwards.columns and 'ict_index_5g_avg' in forwards.columns:
    forwards['goals_ict_index_5g'] = forwards['goals_5g_avg'] * forwards['ict_index_5g_avg']

# Goalkeepers
if 'saves_3g_avg' in goalkeepers.columns and 'clean_sheets_3g_avg' in goalkeepers.columns:
    goalkeepers['saves_clean_sheets_3g'] = goalkeepers['saves_3g_avg'] * goalkeepers['clean_sheets_3g_avg']
if 'expected_goals_conceded_3g_avg' in goalkeepers.columns and 'saves_3g_avg' in goalkeepers.columns:
    goalkeepers['expected_goals_conceded_saves_3g'] = goalkeepers['expected_goals_conceded_3g_avg'] * goalkeepers['saves_3g_avg']

if 'saves_5g_avg' in goalkeepers.columns and 'clean_sheets_5g_avg' in goalkeepers.columns:
    goalkeepers['saves_clean_sheets_5g'] = goalkeepers['saves_5g_avg'] * goalkeepers['clean_sheets_5g_avg']
if 'expected_goals_conceded_5g_avg' in goalkeepers.columns and 'saves_5g_avg' in goalkeepers.columns:
    goalkeepers['expected_goals_conceded_saves_5g'] = goalkeepers['expected_goals_conceded_5g_avg'] * goalkeepers['saves_5g_avg']

# Update feature lists to include new interaction terms, both per-90 and 3g/5g interaction terms
goalkeeper_features = [
    'starts', 'clean_sheets', 'form', 'clean_sheet_probability', 'saves_per_game', 
    'last_season_penalties_saved', 'last_season_clean_sheet_probability', 'saves_clean_sheets_3g', 
    'expected_goals_conceded_saves_3g', 'saves_clean_sheets_5g', 'expected_goals_conceded_saves_5g', 
    'saves_per_90', 'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty',
    'clean_sheets_3g_avg', 'clean_sheets_5g_avg', 'saves_3g_avg', 'saves_5g_avg', 
    'goals_conceded_3g_avg', 'goals_conceded_5g_avg', 'expected_goals_conceded_3g_avg', 'expected_goals_conceded_5g_avg',
    'next_week_specific_fixture_difficulty_3g_avg', 'next_week_specific_fixture_difficulty_5g_avg', 
    'next_week_holistic_fixture_difficulty_3g_avg', 'next_week_holistic_fixture_difficulty_5g_avg', 'bonus_3g_avg', 'bonus_5g_avg'
]

defender_features = [
    'starts', 'clean_sheet_probability', 'ict_index_clean_sheets_3g', 'ict_index_clean_sheets_5g', 'form', 
    'xG&A_form', 'minutes_per_game', 'last_season_xG', 'last_season_xA', 'clean_sheets_goals_3g', 
    'clean_sheets_goals_5g', 'xG_expected_goals_conceded_3g', 'xG_expected_goals_conceded_5g', 
    'clean_sheets_per_90', 'goals_per_90', 'xG_per_90', 'assists_per_90', 'ict_index_per_90',
    'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty', 
    'clean_sheets_3g_avg', 'clean_sheets_5g_avg', 'goals_3g_avg', 'goals_5g_avg', 'xG_3g_avg', 'xG_5g_avg', 
    'goals_conceded_3g_avg', 'goals_conceded_5g_avg', 'bonus_3g_avg', 'bonus_5g_avg',
    'next_week_specific_fixture_difficulty_3g_avg', 'next_week_specific_fixture_difficulty_5g_avg', 
    'next_week_holistic_fixture_difficulty_3g_avg', 'next_week_holistic_fixture_difficulty_5g_avg'
]

midfielder_features = [
    'goal_stat', 'assist_stat', 'key_passes_assists_3g', 'key_passes_assists_5g', 'shots_goals_3g', 
    'shots_goals_5g', 'form', 'xG&A_form', 'minutes_per_game', 'next_week_specific_fixture_difficulty', 
    'next_week_holistic_fixture_difficulty', 'goals_3g_avg', 'goals_5g_avg', 'assists_3g_avg', 'assists_5g_avg', 
    'form_3g_avg', 'form_5g_avg', 'xG_3g_avg', 'xG_5g_avg', 'xA_3g_avg', 'xA_5g_avg', 
    'goals_per_90', 'assists_per_90', 'key_passes_per_90', 'shots_per_90',
    'next_week_specific_fixture_difficulty_3g_avg', 'next_week_specific_fixture_difficulty_5g_avg', 
    'next_week_holistic_fixture_difficulty_3g_avg', 'next_week_holistic_fixture_difficulty_5g_avg', 'bonus_3g_avg', 'bonus_5g_avg'
]

forward_features = [
    'form', 'xG&A_form', 'goals_ict_index_3g', 'goals_ict_index_5g', 'minutes_per_game', 'goal_stat', 'assist_stat', 
    'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty', 'goals_3g_avg', 'goals_5g_avg', 
    'assists_3g_avg', 'assists_5g_avg', 'form_3g_avg', 'form_5g_avg', 'xG_3g_avg', 'xG_5g_avg', 'xA_3g_avg', 'xA_5g_avg', 
    'goals_per_90', 'assists_per_90', 'ict_index_per_90', 
    'next_week_specific_fixture_difficulty_3g_avg', 'next_week_specific_fixture_difficulty_5g_avg', 
    'next_week_holistic_fixture_difficulty_3g_avg', 'next_week_holistic_fixture_difficulty_5g_avg', 'bonus_3g_avg', 'bonus_5g_avg'
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
    rf_model = RandomForestRegressor(n_estimators=250, random_state=42)
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
import numpy as np
import pandas as pd

def apply_trained_model(model, features, csv_file):
    """
    Apply a trained model to a CSV file, make predictions, and save the results.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Verify and log available features in the CSV
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    # Log missing and used features
    if missing_features:
        print(f"Missing features in {csv_file}: {missing_features}")
    if available_features != features:
        print(f"Only using available features for prediction in {csv_file}: {available_features}")


    # Convert available features to numeric, handling non-numeric values by setting them to NaN
    df[available_features] = df[available_features].apply(pd.to_numeric, errors='coerce')
    df[available_features].replace([np.inf, -np.inf], np.nan, inplace=True)
    df[available_features].fillna(0, inplace=True)


    # Proceed with prediction only if all required features are present
    if not missing_features:
        # Predict next week's points using the available features
        X = df[available_features]
        df['predicted_next_week_points'] = model.predict(X)
        
        # Save the predictions to the CSV
        df.to_csv(csv_file, index=False)
        print(f"Predictions saved to {csv_file}")
        
        # Retain only essential columns
        retain_columns(csv_file)
    else:
        print(f"Missing essential features in {csv_file}. Skipping prediction.")


# Apply models to prediction data in prediction_data/2024-25 folder
folder_to_apply='GW10-12'
apply_trained_model(rf_gk_model, goalkeeper_features, 'prediction_data/2024-25/%s/gk.csv'%(folder_to_apply))
apply_trained_model(rf_def_model, defender_features, 'prediction_data/2024-25/%s/def.csv'%(folder_to_apply))
apply_trained_model(rf_mid_model, midfielder_features, 'prediction_data/2024-25/%s/mid.csv'%(folder_to_apply))
apply_trained_model(rf_fwd_model, forward_features, 'prediction_data/2024-25/%s/fwd.csv'%(folder_to_apply))


# Function to adjust predictions based on player availability
def adjust_predictions_by_availability(prediction_csv):
    """
    Adjust the 'predicted_next_week_points' in the prediction CSV based on player availability.
    """
    # Load current availability data
    availability_df = pd.read_csv('current_availability.csv')

    # Ensure required columns are present
    if 'full_name' not in availability_df.columns or 'chance_of_playing_next_round' not in availability_df.columns or 'status' not in availability_df.columns:
        print("The current_availability.csv file is missing required columns.")
        return

    # Load prediction data
    prediction_df = pd.read_csv(prediction_csv)

    # Check if predicted_next_week_points column exists
    if 'predicted_next_week_points' not in prediction_df.columns:
        print(f"{prediction_csv} is missing the 'predicted_next_week_points' column.")
        return

    # Adjust predictions based on availability
    for idx, row in prediction_df.iterrows():
        player_name = row['name']

        # Find closest match in the availability file
        closest_match = process.extractOne(player_name, availability_df['full_name'], score_cutoff=60)
        
        # Ensure a match was found
        if closest_match is not None:
            match, score = closest_match[:2]

            # Retrieve the matched row in the availability data
            matched_row = availability_df.loc[availability_df['full_name'] == match]

            # Check if the status is 'available', handling NaN or non-string values
            status = matched_row['status'].values[0]
            if isinstance(status, str) and status.lower() == 'available':
                chance_of_playing = 100.0
            else:
                # Use the chance of playing from 'chance_of_playing_next_round'
                chance_of_playing = matched_row['chance_of_playing_next_round'].values[0]
            
            print(f"{player_name} has {chance_of_playing} chance of playing based on status '{status}'")
            
            # Adjust predicted points
            adjusted_points = row['predicted_next_week_points'] * (chance_of_playing / 100)
            prediction_df.at[idx, 'predicted_next_week_points'] = round(adjusted_points, 2)
        else:
            print(f"No suitable match found for '{player_name}' in {prediction_csv}")

    # Save adjusted predictions back to the CSV
    prediction_df.to_csv(prediction_csv, index=False)
    print(f"Adjusted predictions saved to {prediction_csv}")

# Apply adjustments to each prediction CSV file
folder_to_apply = 'GW10-12'
adjust_predictions_by_availability(f'prediction_data/2024-25/{folder_to_apply}/gk.csv')
adjust_predictions_by_availability(f'prediction_data/2024-25/{folder_to_apply}/def.csv')
adjust_predictions_by_availability(f'prediction_data/2024-25/{folder_to_apply}/mid.csv')
adjust_predictions_by_availability(f'prediction_data/2024-25/{folder_to_apply}/fwd.csv')



# Define a function to visualize correlation matrix and identify high-correlation pairs
'''
def check_multicollinearity(df, features, position):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
    plt.title(f"{position} Feature Correlation Matrix")
    plt.show()

    # Identify pairs with high correlation (above 0.8)
    corr_matrix = df[features].corr().abs()
    high_corr_pairs = [(col, row) for col in corr_matrix.columns for row in corr_matrix.index
                       if col != row and corr_matrix.loc[col, row] > 0.8]
    
    if high_corr_pairs:
        print(f"High correlation pairs in {position}: {high_corr_pairs}")
    else:
        print(f"No significant multicollinearity in {position}")

# Check for multicollinearity in each dataset
check_multicollinearity(goalkeepers, goalkeeper_features, "Goalkeepers")
check_multicollinearity(defenders, defender_features, "Defenders")
check_multicollinearity(midfielders, midfielder_features, "Midfielders")
check_multicollinearity(forwards, forward_features, "Forwards")
'''
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

def print_feature_importance(model, features, label):
    """
    Print feature importance for a trained Random Forest model as fractions.
    """
    # Get feature importance from the model
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    # Normalize the importance to sum up to 1
    feature_importance_df['Importance'] = feature_importance_df['Importance'] / feature_importance_df['Importance'].sum()
    
    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Print feature importance as fractions
    print(f"\n{label} - Feature Importances (as fractions):")
    for _, row in feature_importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

# Print feature importances for each model as fractions
print_feature_importance(rf_gk_model, goalkeeper_features, "Goalkeepers")
print_feature_importance(rf_def_model, defender_features, "Defenders")
print_feature_importance(rf_mid_model, midfielder_features, "Midfielders")
print_feature_importance(rf_fwd_model, forward_features, "Forwards")
