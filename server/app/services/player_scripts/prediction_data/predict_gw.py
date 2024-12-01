import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, make_scorer, fbeta_score
from sklearn.feature_selection import SelectFromModel, RFECV
import seaborn as sns
import matplotlib.pyplot as plt
import os
from fuzzywuzzy import process
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Global Variables ---
GOALKEEPER_FEATURES = [
    'starts', 'clean_sheets', 'form', 'clean_sheet_probability', 'saves_per_game',
    'last_season_clean_sheet_probability', 'saves_clean_sheets_3g', 'expected_goals_conceded_saves_3g',
    'saves_clean_sheets_5g', 'expected_goals_conceded_saves_5g', 'saves_per_90',
    'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty', 'clean_sheets_3g_avg',
    'clean_sheets_5g_avg', 'saves_3g_avg', 'saves_5g_avg', 'goals_conceded_3g_avg', 'goals_conceded_5g_avg',
    'expected_goals_conceded_3g_avg', 'expected_goals_conceded_5g_avg', 'bonus_3g_avg', 'bonus_5g_avg',
    'clean_sheets_ema', 'saves_ema', 'expected_goals_conceded_ema', 'expected_goals_conceded_momentum_3g',
    'expected_goals_conceded_momentum_5g', 'saves_momentum_3g', 'saves_momentum_5g',
    'recent_minutes_3g', 'recent_minutes_5g', 'adjusted_fixture_difficulty'
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
    'form_difficulty_3g', 'clean_sheets_form_interaction', 'recent_minutes_3g', 'recent_minutes_5g',
    'adjusted_fixture_difficulty'
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
    'xG_momentum_5g', 'xA_momentum_3g', 'xA_momentum_5g', 'form_difficulty_3g', 'form_consistency',
    'goal_contribution', 'assist_contribution', 'recent_minutes_3g', 'recent_minutes_5g',
    'adjusted_fixture_difficulty', 'points_per_minute_delta'
]

FORWARD_FEATURES = [
    'form', 'xG&A_form', 'goals_ict_index_3g', 'goals_ict_index_5g', 'minutes_per_game', 'goal_stat', 'assist_stat',
    'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty', 'goals_3g_avg', 'goals_5g_avg',
    'assists_3g_avg', 'assists_5g_avg', 'form_3g_avg', 'form_5g_avg', 'xG_3g_avg', 'xG_5g_avg', 'xA_3g_avg', 'xA_5g_avg',
    'goals_per_90', 'assists_per_90', 'ict_index_per_90', 'xG_per_90', 'xA_per_90', 'next_week_specific_fixture_difficulty_3g_avg',
    'next_week_specific_fixture_difficulty_5g_avg', 'next_week_holistic_fixture_difficulty_3g_avg',
    'next_week_holistic_fixture_difficulty_5g_avg', 'bonus_3g_avg', 'bonus_5g_avg', 'xG_ema', 'xA_ema', 'ict_index_ema',
    'bonus_ema', 'form_ema', 'goals_ema', 'assists_ema', 'form_momentum_3g', 'form_momentum_5g', 'xG_momentum_3g',
    'xG_momentum_5g', 'xA_momentum_3g', 'xA_momentum_5g', 'form_difficulty_3g', 'form_consistency',
    'goal_contribution', 'assist_contribution', 'recent_minutes_3g', 'recent_minutes_5g',
    'adjusted_fixture_difficulty', 'points_per_minute_delta'
]


TARGET = 'next_week_points'

'''
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
    """
    Check for multicollinearity among features and plot the correlation matrix.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features)

    # Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(scaled_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{position} Feature Correlation Matrix (Scaled)")
    plt.show()

    # Log highly correlated pairs
    corr_matrix = scaled_df.corr().abs()
    high_corr_pairs = [
        (col, row) for col in corr_matrix.columns for row in corr_matrix.index
        if col != row and corr_matrix.loc[col, row] > 0.8
    ]
    if high_corr_pairs:
        print(f"High correlation pairs in {position}: {high_corr_pairs}")
    else:
        print(f"No significant multicollinearity in {position}")

def remove_highly_correlated_features(df, feature_list, threshold=0.8):
    """
    Remove features with high correlations from the feature list.
    """
    corr_matrix = df[feature_list].corr().abs()
    upper_tri = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))

    # Find features to drop
    features_to_drop = set()
    for col in upper_tri.columns:
        for row in upper_tri.index:
            if upper_tri.loc[row, col] > threshold:
                features_to_drop.add(row)

    print(f"Features to drop: {features_to_drop}")
    return [feature for feature in feature_list if feature not in features_to_drop]
'''


def hyperparameter_tune_model(X_train, y_train, model='xgb', task='regression', corr_threshold=0.8):
    """
    Hyperparameter tuning with preprocessing for multicollinearity removal, task-specific feature selection,
    and customized adjustments for classification and regression.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model (str): Model to use ('xgb' or 'rf').
        task (str): Task type ('classification' or 'regression').
        corr_threshold (float): Threshold for removing multicollinear features.

    Returns:
        Best estimator from RandomizedSearchCV with prediction adjustments for regression tasks.
    """

    def preprocess_features(df, features, threshold):
        """
        Check for multicollinearity and remove highly correlated features.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])
        scaled_df = pd.DataFrame(scaled_data, columns=features)
        corr_matrix = scaled_df.corr().abs()
        upper_tri = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
        features_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        return [feature for feature in features if feature not in features_to_drop]

    def select_features_with_rfe(model, X, y, scoring_metric):
        """
        Perform Recursive Feature Elimination (RFE) with Cross-Validation.
        """
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        rfecv = RFECV(estimator=model, step=1, scoring=scoring_metric, cv=kfold, n_jobs=-1, min_features_to_select=10)
        rfecv.fit(X, y)
        return list(X.columns[rfecv.support_])

    # Preprocess features
    initial_features = list(X_train.columns)
    cleaned_features = preprocess_features(X_train, initial_features, corr_threshold)
    X_train_cleaned = X_train[cleaned_features]

    # Task-specific configurations
    if task == 'classification':
        scoring_metric = make_scorer(fbeta_score, beta=1)  # Prioritize recall (reduce false negatives)
        base_model = XGBClassifier(random_state=42)
    elif task == 'regression':
        scoring_metric = 'neg_mean_squared_error'
        base_model = XGBRegressor(random_state=42)
    else:
        raise ValueError(f"Unsupported task type: {task}")

    final_features = select_features_with_rfe(base_model, X_train_cleaned, y_train, scoring_metric)
    X_train_final = X_train_cleaned[final_features]
    expected_features = X_train_final.columns

    # Define model-specific pipeline and parameter grid
    pipeline = None
    if model == 'xgb':
        if task == 'classification':
            model_class = XGBClassifier
            param_grid = {
                'xgb__n_estimators': [100, 300, 500],
                'xgb__max_depth': [3, 6, 9],
                'xgb__learning_rate': [0.01, 0.05, 0.1],
                'xgb__subsample': [0.8, 1.0],
                'xgb__colsample_bytree': [0.8, 1.0],
            }
        elif task == 'regression':
            model_class = XGBRegressor
            param_grid = {
                'xgb__n_estimators': [100, 300, 500],
                'xgb__max_depth': [3, 6, 9],
                'xgb__learning_rate': [0.01, 0.05, 0.1],
                'xgb__subsample': [0.7, 0.8, 1.0],
                'xgb__colsample_bytree': [0.6, 0.8, 1.0],
                'xgb__alpha': [0.9],  # Quantile regression for the 90th percentile
            }
        pipeline = Pipeline([('xgb', model_class(objective='reg:absoluteerror', random_state=42))])
    elif model == 'rf':
        model_class = RandomForestClassifier if task == 'classification' else RandomForestRegressor
        param_grid = {
            'rf__n_estimators': [100, 300, 500],
            'rf__max_depth': [10, 20, None],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
        }
        pipeline = Pipeline([('scaler', StandardScaler()), ('rf', model_class(random_state=42))])
    else:
        raise ValueError(f"Unsupported model type: {model}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipeline, param_grid, scoring=scoring_metric, cv=kfold, n_jobs=-1, random_state=42)
    search.fit(X_train_final, y_train)

    best_model = search.best_estimator_.named_steps[model]

    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({'Feature': final_features, 'Importance': best_model.feature_importances_})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10], align='center')
        plt.gca().invert_yaxis()
        plt.title(f"Top 10 Features by Importance for {task.capitalize()}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        #plt.show()

    # Add range-based scaling for regression
    if task == 'regression':
        X_train_aligned = X_train_final.reindex(columns=expected_features, fill_value=0)
        y_val_preds = pd.Series(search.predict(X_train_aligned), index=X_train_final.index)

        n_ranges = 10
        quantiles = np.linspace(0, 1, n_ranges + 1)
        range_bounds = y_val_preds.quantile(quantiles).values
        ranges = [(range_bounds[i], range_bounds[i + 1]) for i in range(len(range_bounds) - 1)]

        scaling_factors = {}
        for r in ranges:
            range_indices = (y_val_preds >= r[0]) & (y_val_preds < r[1])
            player_count = range_indices.sum()
            if player_count > 0:
                true_values = y_train[range_indices]
                pred_values = y_val_preds[range_indices]
                true_mean = true_values.mean()
                pred_mean = pred_values.mean()
                scaling_factors[r] = true_mean / pred_mean if pred_mean > 0 else 1

        print(f"Computed Scaling Factors: {scaling_factors}")

        def predict_scaled(pipeline, X, expected_features, scaling_factors):
            X_aligned = X.reindex(columns=expected_features, fill_value=0)
            raw_preds = pipeline.predict(X_aligned)
            scaled_preds = np.zeros_like(raw_preds)

            for r, factor in scaling_factors.items():
                range_indices = (raw_preds >= r[0]) & (raw_preds < r[1])
                if range_indices.sum() > 0:
                    scaled_preds[range_indices] = raw_preds[range_indices] * factor

            return scaled_preds

        search.best_estimator_.predict_scaled = lambda X: predict_scaled(
            search.best_estimator_,
            X,
            expected_features=expected_features,
            scaling_factors=scaling_factors,
        )

    return search.best_estimator_



def evaluate_model(model, X_test, y_test, label):
    """
    Evaluate a model using MSE, MAE, and residual plot.
    """
    # Extract expected feature names from the trained model
    if hasattr(model, 'named_steps'):
        if 'xgb' in model.named_steps:
            expected_features = model.named_steps['xgb'].get_booster().feature_names
        elif 'rf' in model.named_steps:
            expected_features = X_test.columns.tolist()  # RandomForest doesn't enforce feature order
        else:
            raise ValueError("Unsupported model type in pipeline.")
    else:
        expected_features = X_test.columns.tolist()  # For standalone models

    # Align X_test with the features expected by the model
    X_test_aligned = X_test.reindex(columns=expected_features, fill_value=0)

    # Predict and evaluate
    predictions = model.predict(X_test_aligned)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"{label} MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Residual Plot
    residuals = y_test - predictions
    sns.histplot(residuals, kde=True)
    plt.title(f'{label} - Residual Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    #plt.show()

# Apply trained model to prediction data and save results
def apply_trained_model(clf_model, reg_model, features, csv_file, confidence_threshold=0.5):
    """
    Apply trained classification and regression models to a CSV file,
    add the 'predicted_next_week_points' column, and retain only the required columns.
    """
    print(f"\nProcessing file: {csv_file}...")

    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} rows from {csv_file}.")

        # Check for duplicate columns and remove them
        duplicate_columns = df.columns[df.columns.duplicated()]
        if not duplicate_columns.empty:
            print(f"Duplicate columns found in {csv_file}: {duplicate_columns.tolist()} - Removing duplicates.")
        df = df.loc[:, ~df.columns.duplicated()]

        # Verify available features
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            print(f"No valid features found in {csv_file}. Skipping prediction.")
            return

        # Convert features to numeric
        df[available_features] = df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)

        # Classification: Predict zero vs. non-zero
        clf_expected_features = clf_model.named_steps['xgb'].get_booster().feature_names
        X_clf = df.reindex(columns=clf_expected_features, fill_value=0)
        X_clf = clean_features(X_clf)
        df["is_predicted_zero"] = clf_model.predict(X_clf)

        # Regression: Predict next week's points for non-zero rows
        non_zero_indices = df[df["is_predicted_zero"] == 0].index
        if not non_zero_indices.empty:
            reg_expected_features = reg_model.named_steps['xgb'].get_booster().feature_names
            X_reg = df.reindex(columns=reg_expected_features, fill_value=0).loc[non_zero_indices]
            X_reg = clean_features(X_reg)
            df.loc[non_zero_indices, "predicted_next_week_points"] = reg_model.predict_scaled(X_reg)
        else:
            print("No non-zero predictions were made.")
            df["predicted_next_week_points"] = 0  # Default to 0 if no predictions are possible

        # Set zero predictions explicitly
        zero_indices = df[df["is_predicted_zero"] == 1].index
        df.loc[zero_indices, "predicted_next_week_points"] = 0
        print(f"Processed predictions: {len(non_zero_indices)} non-zero, {len(zero_indices)} zero.")

        # Save updated DataFrame directly to the same file
        df.to_csv(csv_file, index=False)
        print(f"Updated predictions saved to {csv_file}.")

        # Retain only required columns
        #retain_columns(csv_file)

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")



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
    required_columns = ['name', 'team', 'price', 'form', 'selected', 'predicted_next_week_points']
    available_columns = [col for col in required_columns if col in df.columns]
    
    if len(available_columns) == len(required_columns):
        # Retain only the specified columns
        df = df[required_columns]
        df.to_csv(csv_file, index=False)
        print(f"Updated {csv_file} with only the required columns.")
    else:
        missing_cols = set(required_columns) - set(available_columns)
        print(f"Missing columns in {csv_file}: {missing_cols}")

def engineer_features(df, position):
    """
    Apply feature engineering to the given DataFrame based on the player's position.
    Handles rolling averages, EMA, momentum indicators, interaction terms, and others.
    """

    # --- Universal Features ---
    if 'minutes' in df.columns:
        df['recent_minutes_3g'] = df.groupby('name')['minutes'].transform(lambda x: x.rolling(3, min_periods=1).sum())
        df['recent_minutes_5g'] = df.groupby('name')['minutes'].transform(lambda x: x.rolling(5, min_periods=1).sum())
    if 'next_week_specific_fixture_difficulty' in df.columns and 'next_week_holistic_fixture_difficulty' in df.columns:
        df['adjusted_fixture_difficulty'] = (
            df['next_week_specific_fixture_difficulty'] * 0.5 +
            df['next_week_holistic_fixture_difficulty'] * 0.5
        )

    # --- Rolling Averages, EMA, Momentum ---
    rolling_features = {
        "goalkeepers": ['clean_sheets', 'saves', 'goals_conceded', 'expected_goals_conceded', 'bonus'],
        "defenders": ['clean_sheets', 'form', 'xA', 'xG', 'expected_goals_conceded', 'ict_index', 'bonus', 'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty'],
        "midfielders": ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus', 'key_passes', 'shots', 'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty'],
        "forwards": ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus', 'shots', 'next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty']
    }

    if position.lower() in rolling_features:
        features_to_process = rolling_features[position.lower()]
        for feature in features_to_process:
            if feature in df.columns:
                df[f'{feature}_3g_avg'] = df.groupby('name')[feature].transform(lambda x: x.fillna(0).rolling(3, min_periods=1).mean())
                df[f'{feature}_5g_avg'] = df.groupby('name')[feature].transform(lambda x: x.fillna(0).rolling(5, min_periods=1).mean())
                df[f'{feature}_ema'] = df.groupby('name')[feature].transform(lambda x: x.ewm(span=5, adjust=False).mean())
                df[f'{feature}_momentum_3g'] = df[feature] - df[f'{feature}_3g_avg']
                df[f'{feature}_momentum_5g'] = df[feature] - df[f'{feature}_5g_avg']

    # --- Interaction Terms ---
    interaction_features = {
        "goalkeepers": [
            ('saves_3g_avg', 'clean_sheets_3g_avg', 'saves_clean_sheets_3g'),
            ('expected_goals_conceded_3g_avg', 'saves_3g_avg', 'expected_goals_conceded_saves_3g'),
            ('saves_5g_avg', 'clean_sheets_5g_avg', 'saves_clean_sheets_5g'),
            ('expected_goals_conceded_5g_avg', 'saves_5g_avg', 'expected_goals_conceded_saves_5g')
        ],
        "defenders": [
            ('clean_sheets_3g_avg', 'expected_goals_conceded_3g_avg', 'clean_sheets_expected_goals_conceded_3g'),
            ('clean_sheets_5g_avg', 'expected_goals_conceded_5g_avg', 'clean_sheets_expected_goals_conceded_5g'),
            ('form_3g_avg', 'next_week_specific_fixture_difficulty', 'form_difficulty_3g'),
            ('clean_sheets_3g_avg', 'form_3g_avg', 'clean_sheets_form_interaction')
        ],
        "midfielders": [
            ('key_passes_3g_avg', 'assists_3g_avg', 'key_passes_assists_3g'),
            ('shots_3g_avg', 'goals_3g_avg', 'shots_goals_3g'),
            ('key_passes_5g_avg', 'assists_5g_avg', 'key_passes_assists_5g'),
            ('shots_5g_avg', 'goals_5g_avg', 'shots_goals_5g'),
            ('form_3g_avg', 'next_week_specific_fixture_difficulty', 'form_difficulty_3g')
        ],
        "forwards": [
            ('goals_3g_avg', 'ict_index_3g_avg', 'goals_ict_index_3g'),
            ('goals_5g_avg', 'ict_index_5g_avg', 'goals_ict_index_5g'),
            ('form_3g_avg', 'next_week_specific_fixture_difficulty', 'form_difficulty_3g')
        ]
    }

    if position.lower() in interaction_features:
        for f1, f2, result in interaction_features[position.lower()]:
            if f1 in df.columns and f2 in df.columns:
                df[result] = df[f1] * df[f2]

    # --- Position-Specific Features ---
    if position.lower() in ["midfielders", "forwards"]:
        df['goal_stat'] = 0
        df['assist_stat'] = 0

        if 'last_season_goals' in df.columns and 'last_season_xG' in df.columns:
            df['goal_stat'] = df['goal_stat'] + df['last_season_goals'].fillna(0) + df['last_season_xG'].fillna(0)
        if 'last_season_assists' in df.columns and 'last_season_xA' in df.columns:
            df['assist_stat'] = df['assist_stat'] + df['last_season_assists'].fillna(0) + df['last_season_xA'].fillna(0)
        df['form_consistency'] = df.groupby('name')['form'].transform(lambda x: x.rolling(5, min_periods=1).std())
        df['goal_contribution'] = df['goals'] / (df['total_points'] + 1)
        df['assist_contribution'] = df['assists'] / (df['total_points'] + 1)

        df['points_per_minute_delta'] = 0
        if 'last_season_points_per_minute' in df.columns: 
            df['points_per_minute_delta'] = (
                (df['total_points'] / df['minutes'].replace(0, 1)) - df['last_season_points_per_minute']
            )

    # --- Per-90 Metrics ---
    per_90_features = {
        "goalkeepers": ['saves'],
        "defenders": ['clean_sheets', 'xA', 'xG', 'ict_index'],
        "midfielders": ['goals', 'assists', 'key_passes', 'shots', 'xG', 'xA', 'ict_index'],
        "forwards": ['goals', 'assists', 'xG', 'xA', 'ict_index']
    }

    if position.lower() in per_90_features:
        df['cumulative_minutes'] = df.groupby('name')['minutes'].cumsum()
        for feature in per_90_features[position.lower()]:
            if feature in df.columns:
                df[f'{feature}_per_90'] = np.where(
                    df['cumulative_minutes'] > 0,
                    df.groupby('name')[feature].cumsum() / (df['cumulative_minutes'] / 90),
                    0
                )
        df.drop(columns=['cumulative_minutes'], inplace=True)

    return df

def clean_target(y):
    """
    Clean target variable by handling missing or invalid values.
    """
    # Replace infinite values with NaN
    y = y.replace([np.inf, -np.inf], np.nan)

    # Ensure target is numeric and handle missing values
    y = pd.to_numeric(y, errors='coerce')

    # Fill remaining NaNs with a default value (e.g., 0 for log-transformed data)
    y = y.fillna(0)

    return y

def clean_features(X):
    """
    Clean feature matrix by handling missing values and ensuring numeric data.
    """
    # Replace infinite values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Convert all columns to numeric types, coercing errors to NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Fill NaN values with a default value (e.g., 0)
    X = X.fillna(0)

    return X



def predict_gw(gw_dir):
    """
    Train models on training data and apply predictions for a specific game week.
    """
    global GOALKEEPER_FEATURES, DEFENDER_FEATURES, MIDFIELDER_FEATURES, FORWARD_FEATURES

    # Feature Engineering and Training Data Loading
    train_data_files = {
        "goalkeepers": ("train_data/goalkeeper.csv", GOALKEEPER_FEATURES),
        "defenders": ("train_data/defender.csv", DEFENDER_FEATURES),
        "midfielders": ("train_data/midfielder.csv", MIDFIELDER_FEATURES),
        "forwards": ("train_data/forward.csv", FORWARD_FEATURES),
    }

    models = {}

    for position, (file_path, features) in train_data_files.items():
        print(f"\nProcessing training data for {position.capitalize()}...")

        try:
            # Load Training Data
            df = pd.read_csv(file_path)

            # Feature Engineering
            df = engineer_features(df, position)

            # Classification Step: Zero vs Non-Zero
            df['is_zero'] = (df[TARGET] <= 0).astype(int)
            X_clf, y_clf = df[features], df['is_zero']
            X_clf = clean_features(X_clf)
            y_clf = clean_target(y_clf)
            X_clf, y_clf = X_clf.align(y_clf, axis=0, join='inner')

            if X_clf.empty or y_clf.empty:
                print(f"No data available for classification for {position}. Skipping...")
                continue

            # Train-Test Split for Classification
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
                X_clf, y_clf, test_size=0.2, random_state=42
            )
            clf_pipeline = hyperparameter_tune_model(X_train_clf, y_train_clf, model="xgb", task="classification")
            evaluate_model(clf_pipeline, X_test_clf, y_test_clf, f"{position.capitalize()} Classifier")

            # Predictions from Classification Model
            expected_features = clf_pipeline.named_steps['xgb'].get_booster().feature_names if 'xgb' in clf_pipeline.named_steps else clf_pipeline.named_steps['rf'].feature_names_in_
            X_clf_aligned = X_clf.reindex(columns=expected_features, fill_value=0)

            df['predicted_non_zero'] = clf_pipeline.predict(X_clf_aligned) == 0

            # Regression Step: Non-Zero Predictions
            non_zero_data = df[df['predicted_non_zero']]
            if non_zero_data.empty:
                print(f"No non-zero data available for {position}. Skipping regression step...")
                continue

            X_reg, y_reg = non_zero_data[features], non_zero_data[TARGET]
            X_reg = clean_features(X_reg)
            X_reg, y_reg = X_reg.align(y_reg, axis=0, join='inner')
            y_reg = clean_target(y_reg)

            # Train-Test Split for Regression
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=42
            )


            reg_pipeline = hyperparameter_tune_model(X_train_reg, y_train_reg, model="xgb", task="regression")

            # Train Regression Model
            expected_features = (
                reg_pipeline.named_steps['xgb'].get_booster().feature_names
                if 'xgb' in reg_pipeline.named_steps
                else reg_pipeline.named_steps['rf'].feature_names_in_
            )
            X_test_reg_aligned = X_test_reg.reindex(columns=expected_features, fill_value=0)

            # Use the aligned data for prediction
            y_pred_reg = reg_pipeline.predict(X_test_reg_aligned)

            evaluate_model(reg_pipeline, X_test_reg_aligned, y_test_reg, f"{position.capitalize()} Regressor")

            # Save Trained Models
            models[position] = (clf_pipeline, reg_pipeline)


        except Exception as e:
            print(f"Error processing training data for {position}: {e}")

    print("\nTraining complete. Proceeding to prediction phase...")


    
    # Step 2: Prediction Phase
    prediction_base_path = "prediction_data/2024-25"
    position_file_map = {
    "goalkeepers": "gk",
    "defenders": "def",
    "midfielders": "mid",
    "forwards": "fwd"
    }

    for gw_dir in os.listdir(prediction_base_path):
        gw_path = os.path.join(prediction_base_path, gw_dir)
        if not os.path.isdir(gw_path):
            continue

        print(f"\nProcessing predictions for game week: {gw_dir}...")

        for position, (file_path, features) in train_data_files.items():
            clf_pipeline, reg_pipeline = models.get(position, (None, None))
            if clf_pipeline is None or reg_pipeline is None:
                print(f"No trained models available for {position.capitalize()}. Skipping...")
                continue

            # Access the correct file name
            file_prefix = position_file_map.get(position.lower())
            prediction_file = f"{gw_path}/{file_prefix}.csv"
            if not os.path.exists(prediction_file):
                print(f"Prediction file not found for {position.capitalize()}: {prediction_file}")
                continue

            try:
                # Load and Apply Feature Engineering
                df = pd.read_csv(prediction_file)
                print(f"Loaded {len(df)} rows from {prediction_file}.")

                # Apply Trained Models
                apply_trained_model(clf_pipeline, reg_pipeline, features, prediction_file)

                # Adjust Predictions
                adjust_predictions_by_availability(prediction_file)
                print(f"Adjusted predictions for {position.capitalize()} in {gw_dir}.")

            except Exception as e:
                print(f"Error applying predictions for {position.capitalize()} in {gw_dir}: {e}")
                continue

    print("\nPrediction phase complete.")
   