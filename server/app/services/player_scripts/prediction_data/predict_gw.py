import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import matplotlib.pyplot as plt
import os
from fuzzywuzzy import process

# --- Global Variables ---
GOALKEEPER_FEATURES = [
    'starts', 'clean_sheets', 'form', 'clean_sheet_probability', 'saves_per_game',
    'last_season_clean_sheet_probability', 'saves_clean_sheets_3g', 'expected_goals_conceded_saves_3g',
    'saves_clean_sheets_5g', 'expected_goals_conceded_saves_5g', 'saves_per_90',
    'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty', 'clean_sheets_3g_avg',
    'clean_sheets_5g_avg', 'saves_3g_avg', 'saves_5g_avg', 'goals_conceded_3g_avg', 'goals_conceded_5g_avg',
    'expected_goals_conceded_3g_avg', 'expected_goals_conceded_5g_avg', 'bonus_3g_avg', 'bonus_5g_avg',
    'clean_sheets_ema', 'saves_ema', 'expected_goals_conceded_ema', 'expected_goals_conceded_momentum_3g',
    'expected_goals_conceded_momentum_5g', 'saves_momentum_3g', 'saves_momentum_5g'
]

DEFENDER_FEATURES = [
    'starts', 'clean_sheet_probability', 'form', 'xG&A_form', 'minutes_per_game', 'last_season_xG',
    'last_season_xA', 'clean_sheets_per_90', 'xG_per_90', 'xA_per_90', 'ict_index_per_90',
    'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty', 'clean_sheets_3g_avg',
    'clean_sheets_5g_avg', 'xA_3g_avg', 'xA_5g_avg', 'xG_3g_avg', 'xG_5g_avg', 'expected_goals_conceded_3g_avg',
    'expected_goals_conceded_5g_avg', 'bonus_3g_avg', 'bonus_5g_avg', 'form_3g_avg', 'form_5g_avg',
    'next_week_specific_fixture_difficulty_3g_avg', 'next_week_specific_fixture_difficulty_5g_avg',
    'next_week_holistic_fixture_difficulty_3g_avg', 'next_week_holistic_fixture_difficulty_5g_avg',
    'clean_sheets_ema', 'expected_goals_conceded_ema', 'xG_ema', 'xA_ema', 'ict_index_ema', 'bonus_ema',
    'form_ema', 'expected_goals_conceded_momentum_3g', 'expected_goals_conceded_momentum_5g', 'form_momentum_3g',
    'form_momentum_5g', 'clean_sheets_expected_goals_conceded_5g', 'clean_sheets_expected_goals_conceded_3g',
    'form_difficulty_3g'
]

MIDFIELDER_FEATURES = [
    'goal_stat', 'assist_stat', 'key_passes_assists_3g', 'key_passes_assists_5g', 'shots_goals_3g', 'shots_goals_5g',
    'form', 'xG&A_form', 'minutes_per_game', 'next_week_specific_fixture_difficulty',
    'next_week_holistic_fixture_difficulty', 'goals_3g_avg', 'goals_5g_avg', 'assists_3g_avg', 'assists_5g_avg',
    'form_3g_avg', 'form_5g_avg', 'xG_3g_avg', 'xG_5g_avg', 'xA_3g_avg', 'xA_5g_avg', 'goals_per_90', 'assists_per_90',
    'key_passes_per_90', 'shots_per_90', 'xG_per_90', 'xA_per_90', 'next_week_specific_fixture_difficulty_3g_avg',
    'next_week_specific_fixture_difficulty_5g_avg', 'next_week_holistic_fixture_difficulty_3g_avg',
    'next_week_holistic_fixture_difficulty_5g_avg', 'bonus_3g_avg', 'bonus_5g_avg', 'xG_ema', 'xA_ema', 'ict_index_ema',
    'bonus_ema', 'form_ema', 'goals_ema', 'assists_ema', 'form_momentum_3g', 'form_momentum_5g', 'xG_momentum_3g',
    'xG_momentum_5g', 'xA_momentum_3g', 'xA_momentum_5g', 'form_difficulty_3g'
]

FORWARD_FEATURES = [
    'form', 'xG&A_form', 'goals_ict_index_3g', 'goals_ict_index_5g', 'minutes_per_game', 'goal_stat', 'assist_stat',
    'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty', 'goals_3g_avg', 'goals_5g_avg',
    'assists_3g_avg', 'assists_5g_avg', 'form_3g_avg', 'form_5g_avg', 'xG_3g_avg', 'xG_5g_avg', 'xA_3g_avg', 'xA_5g_avg',
    'goals_per_90', 'assists_per_90', 'ict_index_per_90', 'xG_per_90', 'xA_per_90', 'next_week_specific_fixture_difficulty_3g_avg',
    'next_week_specific_fixture_difficulty_5g_avg', 'next_week_holistic_fixture_difficulty_3g_avg',
    'next_week_holistic_fixture_difficulty_5g_avg', 'bonus_3g_avg', 'bonus_5g_avg', 'xG_ema', 'xA_ema', 'ict_index_ema',
    'bonus_ema', 'form_ema', 'goals_ema', 'assists_ema', 'form_momentum_3g', 'form_momentum_5g', 'xG_momentum_3g',
    'xG_momentum_5g', 'xA_momentum_3g', 'xA_momentum_5g', 'form_difficulty_3g'
]

TARGET = 'next_week_points'

# --- Utility Functions ---
def print_feature_importance(model, features, label):
    """
    Print feature importances for the model as fractions.
    """
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(f"\n{label} Feature Importances:")
    for _, row in feature_importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

def check_multicollinearity(df, features, position):
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features)

    # Correlation matrix for scaled data
    plt.figure(figsize=(12, 8))
    sns.heatmap(scaled_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{position} Feature Correlation Matrix (Scaled)")
    plt.show()

    # Identify pairs with high correlation (above 0.8)
    corr_matrix = scaled_df.corr().abs()
    high_corr_pairs = [(col, row) for col in corr_matrix.columns for row in corr_matrix.index
                       if col != row and corr_matrix.loc[col, row] > 0.8]
    
    if high_corr_pairs:
        print(f"High correlation pairs in {position}: {high_corr_pairs}")
    else:
        print(f"No significant multicollinearity in {position}")
        
def remove_highly_correlated_features(df, feature_list, threshold=0.8):
    """
    Removes highly correlated features from a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        feature_list (list): List of features to check for correlations.
        threshold (float): Correlation threshold above which features are considered highly correlated.

    Returns:
        list: A pruned list of features with reduced multicollinearity.
    """
    print(f"Original feature count: {len(feature_list)}")

    # Compute the correlation matrix for the feature set
    corr_matrix = df[feature_list].corr().abs()

    # Extract upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(
        ~np.tril(np.ones(corr_matrix.shape)).astype(bool)
    )

    # Find feature pairs with correlation above the threshold
    high_corr_pairs = [(col, row) for col in upper_tri.columns for row in upper_tri.index
                       if col != row and upper_tri.loc[row, col] > threshold]

    # Log highly correlated pairs
    print(f"Highly correlated feature pairs (threshold={threshold}): {high_corr_pairs}")

    # Create a set to hold features to drop
    features_to_drop = set()

    for col, row in high_corr_pairs:
        # Drop the less relevant feature (here, we arbitrarily drop the second feature in the pair)
        features_to_drop.add(row)

    # Log dropped features
    print(f"Features to drop due to high correlation: {features_to_drop}")

    # Return the pruned feature list
    pruned_features = [feature for feature in feature_list if feature not in features_to_drop]
    print(f"Pruned feature count: {len(pruned_features)}")
    return pruned_features

# --- Model Training and Evaluation ---
def hyperparameter_tune_rf(X_train, y_train):
    """
    Hyperparameter tuning for Random Forest Regressor.
    """
    rf_model = RandomForestRegressor(random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', rf_model)
    ])
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['auto', 'sqrt', 'log2']
    }

    search = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    print("Best Parameters:", search.best_params_)
    return search.best_estimator_

def evaluate_model(model, X_test, y_test, label):
    """
    Evaluate a model using Mean Squared Error and Mean Absolute Error.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"{label} MSE: {mse:.4f}, MAE: {mae:.4f}")

# Apply trained model to prediction data and save results
def apply_trained_model(clf_model, reg_model, features, csv_file):
    """
    Apply trained classification and regression models to a CSV file, make predictions, and save the results.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Check for duplicate columns
    duplicate_columns = df.columns[df.columns.duplicated()]
    if not duplicate_columns.empty:
        print(f"Duplicate columns in {csv_file}: {duplicate_columns}")
    
    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Verify and log available features in the CSV
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    
    # Log missing and used features
    if missing_features:
        print(f"Missing features in {csv_file}: {missing_features}")
    if available_features != features:
        print(f"Only using available features for prediction in {csv_file}: {available_features}")

    # Convert available features to numeric, handling non-numeric values by setting them to NaN
    df.loc[:, available_features] = df[available_features].apply(pd.to_numeric, errors='coerce')
    df[available_features] = df[available_features].replace([np.inf, -np.inf], np.nan)
    df[available_features] = df[available_features].fillna(0)

    # Proceed with prediction only if all required features are present
    if not missing_features:
        # Predict zero vs. non-zero using the classifier
        X = df[available_features]
        df['is_predicted_zero'] = clf_model.predict(X)

        # Predict next week's points for non-zero rows using the regressor
        non_zero_indices = df[df['is_predicted_zero'] == 0].index
        df.loc[non_zero_indices, 'predicted_next_week_points'] = reg_model.predict(X.loc[non_zero_indices])

        # Set predicted points to 0 for rows classified as zero
        zero_indices = df[df['is_predicted_zero'] == 1].index
        df.loc[zero_indices, 'predicted_next_week_points'] = 0

        # Save the predictions to the CSV
        df.to_csv(csv_file, index=False)
        print(f"Predictions saved to {csv_file}")
        retain_columns(csv_file)
    else:
        print(f"Missing essential features in {csv_file}. Skipping prediction.")


def adjust_predictions_by_availability(prediction_csv):
    """
    Adjust the 'predicted_next_week_points' in the prediction CSV based on player availability.
    """
    # Load current availability data
    availability_df = pd.read_csv('current_availability.csv')

    # Ensure required columns are present
    required_columns = {'full_name', 'chance_of_playing_next_round', 'status'}
    if not required_columns.issubset(availability_df.columns):
        print("The current_availability.csv file is missing required columns.")
        return

    # Convert 'full_name' column to string and fill NaNs with an empty string
    availability_df['full_name'] = availability_df['full_name'].fillna('').astype(str)

    # Load prediction data
    prediction_df = pd.read_csv(prediction_csv)

    # Check if 'predicted_next_week_points' column exists
    if 'predicted_next_week_points' not in prediction_df.columns:
        print(f"{prediction_csv} is missing the 'predicted_next_week_points' column.")
        return

    # Adjust predictions based on availability
    for idx, row in prediction_df.iterrows():
        player_name = str(row['name'])  # Ensure the name is a string

        # Find the closest match in the availability file
        closest_match = process.extractOne(player_name, availability_df['full_name'], score_cutoff=60)

        if closest_match:
            matched_name = closest_match[0]
            matched_row = availability_df.loc[availability_df['full_name'] == matched_name]

            if not matched_row.empty:
                # Retrieve 'chance_of_playing_next_round' safely
                chance_of_playing = matched_row['chance_of_playing_next_round'].iloc[0]

                # Adjust prediction
                scaled_points = row['predicted_next_week_points'] * (chance_of_playing / 100)
                prediction_df.at[idx, 'predicted_next_week_points'] = round(float(scaled_points), 2)
        else:
            print(f"No match found in the availability data for player: {player_name}")

    # Save adjusted predictions back to the CSV
    prediction_df.to_csv(prediction_csv, index=False)
    print(f"Status-adjusted predictions saved to {prediction_csv}")




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


def predict_gw(gw_dir):
    global GOALKEEPER_FEATURES, DEFENDER_FEATURES, MIDFIELDER_FEATURES, FORWARD_FEATURES
    # Load the training data
    goalkeepers = pd.read_csv('train_data/goalkeeper.csv')
    defenders = pd.read_csv('train_data/defender.csv')
    midfielders = pd.read_csv('train_data/midfielder.csv')
    forwards = pd.read_csv('train_data/forward.csv')


    # Advanced feature engineering
    # Goal and Assist Stats for Midfielders and Forwards
    midfielders['goal_stat'] = midfielders['last_season_goals'] + midfielders['last_season_xG']
    midfielders['assist_stat'] = midfielders['last_season_assists'] + midfielders['last_season_xA']
    forwards['goal_stat'] = forwards['last_season_goals'] + forwards['last_season_xG']
    forwards['assist_stat'] = forwards['last_season_assists'] + forwards['last_season_xA']

    # Moving Averages for Recent Form - 3 and 5 game averages, now including 'bonus'
    for df, features in [(midfielders, ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus', 'key_passes', 'shots']),
                        (forwards, ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus', 'shots', 'key_passes']),
                        (defenders, ['clean_sheets', 'form', 'xA', 'xG', 'expected_goals_conceded', 'ict_index', 'bonus']),
                        (goalkeepers, ['clean_sheets', 'saves', 'goals_conceded', 'expected_goals_conceded', 'bonus'])]:
        for feature in features:
            df[f'{feature}_3g_avg'] = df.groupby('name')[feature].transform(lambda x: x.fillna(0).rolling(3, min_periods=1).mean())
            df[f'{feature}_5g_avg'] = df.groupby('name')[feature].transform(lambda x: x.fillna(0).rolling(5, min_periods=1).mean())

    # Weighted Averages for last 5 games
    # Apply exponential moving average to key features
    for df, features in [(midfielders, ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus']),
                        (forwards, ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus']),
                        (defenders, ['clean_sheets', 'form', 'xG', 'xA', 'ict_index', 'expected_goals_conceded', 'bonus']),
                        (goalkeepers, ['clean_sheets', 'saves', 'expected_goals_conceded'])]:
        for feature in features:
            if feature in df.columns:
                # Calculate EMA with a span of 5
                df[f'{feature}_ema'] = df.groupby('name')[feature].transform(lambda x: x.ewm(span=5, adjust=False).mean())

    # Momentum Indicators based on 3-game averages
    for df, features in [(midfielders, ['form', 'xG', 'xA']), (forwards, ['form', 'xG', 'xA']), (defenders, ['expected_goals_conceded', 'form']), (goalkeepers, ['expected_goals_conceded', 'saves'])]:
        for feature in features: 
            if feature in df.columns:
                # Calculate momentum as the difference from a 3-game average
                df[f'{feature}_momentum_3g'] = df[feature] - df[f'{feature}_3g_avg']
                # Calculate momentum as the difference from a 5-game average
                df[f'{feature}_momentum_5g'] = df[feature] - df[f'{feature}_5g_avg']


    # Per-90 Stats - Calculate Cumulative Per-90 Stats
    for df, feature_list in [(defenders, ['clean_sheets', 'xA', 'xG', 'ict_index']), 
                            (midfielders, ['xG', 'xA', 'goals', 'assists', 'key_passes', 'shots', 'ict_index']),
                            (forwards, ['xG', 'xA', 'goals', 'assists', 'ict_index']),
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
    if 'clean_sheets_3g_avg' in defenders.columns and 'expected_goals_conceded_3g_avg' in defenders.columns:
        defenders['clean_sheets_expected_goals_conceded_3g'] = defenders['clean_sheets_3g_avg'] * defenders['expected_goals_conceded_3g_avg']
    if 'clean_sheets_5g_avg' in defenders.columns and 'expected_goals_conceded_5g_avg' in defenders.columns:
        defenders['clean_sheets_expected_goals_conceded_5g'] = defenders['clean_sheets_5g_avg'] * defenders['expected_goals_conceded_5g_avg']
    if 'form_3g_avg' in defenders.columns and 'next_week_specific_fixture_difficulty' in defenders.columns: 
        defenders['form_difficulty_3g'] = defenders['form_3g_avg'] * defenders['next_week_specific_fixture_difficulty']

    # Midfielders
    if 'key_passes_3g_avg' in midfielders.columns and 'assists_3g_avg' in midfielders.columns:
        midfielders['key_passes_assists_3g'] = midfielders['key_passes_3g_avg'] * midfielders['assists_3g_avg']
    if 'shots_3g_avg' in midfielders.columns and 'goals_3g_avg' in midfielders.columns:
        midfielders['shots_goals_3g'] = midfielders['shots_3g_avg'] * midfielders['goals_3g_avg']
    if 'key_passes_5g_avg' in midfielders.columns and 'assists_5g_avg' in midfielders.columns:
        midfielders['key_passes_assists_5g'] = midfielders['key_passes_5g_avg'] * midfielders['assists_5g_avg']
    if 'shots_5g_avg' in midfielders.columns and 'goals_5g_avg' in midfielders.columns:
        midfielders['shots_goals_5g'] = midfielders['shots_5g_avg'] * midfielders['goals_5g_avg']
    if 'form_3g_avg' in midfielders.columns and 'next_week_specific_fixture_difficulty' in midfielders.columns: 
        midfielders['form_difficulty_3g'] = midfielders['form_3g_avg'] * midfielders['next_week_specific_fixture_difficulty']


    # Forwards
    if 'goals_3g_avg' in forwards.columns and 'ict_index_3g_avg' in forwards.columns:
        forwards['goals_ict_index_3g'] = forwards['goals_3g_avg'] * forwards['ict_index_3g_avg']
    if 'goals_5g_avg' in forwards.columns and 'ict_index_5g_avg' in forwards.columns:
        forwards['goals_ict_index_5g'] = forwards['goals_5g_avg'] * forwards['ict_index_5g_avg']
    if 'form_3g_avg' in forwards.columns and 'next_week_specific_fixture_difficulty' in forwards.columns: 
        forwards['form_difficulty_3g'] = forwards['form_3g_avg'] * forwards['next_week_specific_fixture_difficulty']

    # Goalkeepers
    if 'saves_3g_avg' in goalkeepers.columns and 'clean_sheets_3g_avg' in goalkeepers.columns:
        goalkeepers['saves_clean_sheets_3g'] = goalkeepers['saves_3g_avg'] * goalkeepers['clean_sheets_3g_avg']
    if 'expected_goals_conceded_3g_avg' in goalkeepers.columns and 'saves_3g_avg' in goalkeepers.columns:
        goalkeepers['expected_goals_conceded_saves_3g'] = goalkeepers['expected_goals_conceded_3g_avg'] * goalkeepers['saves_3g_avg']

    if 'saves_5g_avg' in goalkeepers.columns and 'clean_sheets_5g_avg' in goalkeepers.columns:
        goalkeepers['saves_clean_sheets_5g'] = goalkeepers['saves_5g_avg'] * goalkeepers['clean_sheets_5g_avg']
    if 'expected_goals_conceded_5g_avg' in goalkeepers.columns and 'saves_5g_avg' in goalkeepers.columns:
        goalkeepers['expected_goals_conceded_saves_5g'] = goalkeepers['expected_goals_conceded_5g_avg'] * goalkeepers['saves_5g_avg']



    for position, df, features, csv_file in [
            ("Goalkeepers", goalkeepers, GOALKEEPER_FEATURES, f'prediction_data/2024-25/{gw_dir}/gk.csv'),
            ("Defenders", defenders, DEFENDER_FEATURES, f'prediction_data/2024-25/{gw_dir}/def.csv'),
            ("Midfielders", midfielders, MIDFIELDER_FEATURES, f'prediction_data/2024-25/{gw_dir}/mid.csv'),
            ("Forwards", forwards, FORWARD_FEATURES, f'prediction_data/2024-25/{gw_dir}/fwd.csv')
        ]:
        print(f"\nProcessing {position}...")

        # Step 1: Add a binary target column for zero vs. non-zero
        df['is_zero'] = (df[TARGET] == 0).astype(int)

        # Train-test split for classification
        X_clf, y_clf = df[features], df['is_zero']
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42
        )

        # Train a Random Forest Classifier
        print(f"Training classification model for {position}...")
        clf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        clf_pipeline.fit(X_train_clf, y_train_clf)

        # Evaluate the classification model
        y_pred_clf = clf_pipeline.predict(X_test_clf)
        print(f"\n{position} Classification Report:")
        print(classification_report(y_test_clf, y_pred_clf))

        # Display feature importance for the classifier
        clf_model = clf_pipeline.named_steps['clf']
        clf_feature_importances = clf_model.feature_importances_
        print(f"\n{position} Classifier Feature Importances:")
        sorted_clf_features = sorted(
            zip(features, clf_feature_importances), key=lambda x: x[1], reverse=True
        )
        for feature, importance in sorted_clf_features:
            print(f"{feature}: {importance:.4f}")

        # Step 2: Use classifier predictions to filter non-zero rows
        df['predicted_non_zero'] = clf_pipeline.predict(X_clf) == 0
        non_zero_data = df[df['predicted_non_zero']]
        X_reg, y_reg = non_zero_data[features], non_zero_data[TARGET]

        # Train-test split for regression
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        # Train a Random Forest Regressor for non-zero values
        print(f"Training regression model for {position}...")
        reg_pipeline = hyperparameter_tune_rf(X_train_reg, y_train_reg)

        # Display feature importance for the regressor
        reg_model = reg_pipeline.named_steps['rf']
        reg_feature_importances = reg_model.feature_importances_
        print(f"\n{position} Regressor Feature Importances:")
        sorted_reg_features = sorted(
            zip(features, reg_feature_importances), key=lambda x: x[1], reverse=True
        )
        for feature, importance in sorted_reg_features:
            print(f"{feature}: {importance:.4f}")

        # Evaluate regression model
        print(f"\nEvaluating regression model for {position}...")
        y_pred_reg = reg_pipeline.predict(X_test_reg)
        mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
        mae_reg = mean_absolute_error(y_test_reg, y_pred_reg)
        print(f"{position} Regression MSE: {mse_reg:.4f}, MAE: {mae_reg:.4f}")

        # Apply models to prediction data in prediction_data/2024-25 folder
        print(f"Applying trained models to prediction data for {position}...")
        apply_trained_model(clf_pipeline, reg_pipeline, features, csv_file)

        # Apply adjustments to each prediction CSV file
        print(f"Adjusting predictions based on availability for {position}...")
        adjust_predictions_by_availability(csv_file)

    print("\nModel training, prediction, and adjustment complete.")
