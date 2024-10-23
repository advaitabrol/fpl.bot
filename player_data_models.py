import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb



goalkeepers = pd.read_csv('train_data/goalkeeper.csv')
defenders = pd.read_csv('train_data/defender.csv')
midfielders = pd.read_csv('train_data/midfielder.csv')
forwards = pd.read_csv('train_data/forward.csv')


goalkeepers.fillna(0, inplace=True); 
defenders.fillna(0, inplace=True) 
midfielders.fillna(0, inplace=True) 
forwards.fillna(0, inplace=True) 

target = 'next_week_points'

goalkeeper_features = [
    'minutes','goals_conceded','expected_goals_conceded','saves','penalties_saved','total_points','bonus','clean_sheets','xA','starts','form','clean_sheet_probability','last_season_penalties_saved','last_season_expected_goals_conceded','last_season_clean_sheet_probability','next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty', 'saves_per_game','last_season_total_saves'
]
X_gk = goalkeepers[goalkeeper_features]
y_gk = goalkeepers[target]
# Split the data into training and test sets
X_train_gk, X_test_gk, y_train_gk, y_test_gk = train_test_split(X_gk, y_gk, test_size=0.2, random_state=42)


defender_features = [
    'minutes','goals','xG','assists','xA','total_points','shots','key_passes','ict_index','bonus','clean_sheets','goals_conceded','expected_goals_conceded','clean_sheet_probability','last_season_expected_goals_conceded','last_season_clean_sheet_probability','starts','form','xG&A_form','minutes_per_game','last_season_xG','last_season_xA','next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty'
]
X_def = defenders[defender_features]
y_def = defenders[target]
# Split the data into training and test sets
X_train_def, X_test_def, y_train_def, y_test_def = train_test_split(X_def, y_def, test_size=0.2, random_state=42)


midfielder_features = [
    'goals_conceded', 'clean_sheets', 'minutes','goals','xG','assists','xA','total_points','shots','key_passes','ict_index','bonus', 'starts','form','xG&A_form','minutes_per_game','last_season_goals','last_season_assists','last_season_xG','last_season_xA','last_season_points_per_minute','next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty',
]
X_mid = midfielders[midfielder_features]
y_mid = midfielders[target]
# Split the data into training and test sets
X_train_mid, X_test_mid, y_train_mid, y_test_mid = train_test_split(X_mid, y_mid, test_size=0.2, random_state=42)


forward_features = [
    'minutes','goals','xG','assists','xA','total_points','shots','key_passes','ict_index','bonus','starts','form','xG&A_form','minutes_per_game','last_season_goals','last_season_assists','last_season_xG','last_season_xA','last_season_points_per_minute','next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty'
]
X_fwd = forwards[forward_features]
y_fwd = forwards[target]
# Split the data into training and test sets
X_train_fwd, X_test_fwd, y_train_fwd, y_test_fwd = train_test_split(X_fwd, y_fwd, test_size=0.2, random_state=42)

###RANDOM FOREST STUFF


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_gk, y_train_gk)

# Make predictions on the test set
rf_predictions = rf_model.predict(X_test_gk)

# Evaluate the model's performance
rf_mse = mean_squared_error(y_test_gk, rf_predictions)
rf_mae = mean_absolute_error(y_test_gk, rf_predictions)

print(f"Random Forest MSE (GKs): {rf_mse}")
print(f"Random Forest MAE (GKs): {rf_mae}")



rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_def, y_train_def)

# Make predictions on the test set
rf_predictions = rf_model.predict(X_test_def)

# Evaluate the model's performance
rf_mse = mean_squared_error(y_test_def, rf_predictions)
rf_mae = mean_absolute_error(y_test_def, rf_predictions)

print(f"Random Forest MSE (DEFs): {rf_mse}")
print(f"Random Forest MAE (DEFs): {rf_mae}")



rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_mid, y_train_mid)

# Make predictions on the test set
rf_predictions = rf_model.predict(X_test_mid)

# Evaluate the model's performance
rf_mse = mean_squared_error(y_test_mid, rf_predictions)
rf_mae = mean_absolute_error(y_test_mid, rf_predictions)

print(f"Random Forest MSE (MIDs): {rf_mse}")
print(f"Random Forest MAE (MIDs): {rf_mae}")



rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_fwd, y_train_fwd)

# Make predictions on the test set
rf_predictions = rf_model.predict(X_test_fwd)

# Evaluate the model's performance
rf_mse = mean_squared_error(y_test_fwd, rf_predictions)
rf_mae = mean_absolute_error(y_test_fwd, rf_predictions)

print(f"Random Forest MSE (FWDs): {rf_mse}")
print(f"Random Forest MAE (FWDs): {rf_mae}")