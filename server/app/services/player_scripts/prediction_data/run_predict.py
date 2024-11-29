import os
import numpy as np
import pandas as pd
import argparse

from prediction_data.add_prices import add_price_to_prediction_data;
from prediction_data.predict_gw import predict_gw; 
from prediction_data.reformat import merge_player_weeks;
from prediction_data.consolidate import combine_csv_files; 

from prediction_data.predict_gw import engineer_features

def create_prediction_data(base_dir, prediction_dir, gw_folder_name, season='2024-25'):
    """
    Generate prediction data for a specific game week by applying feature engineering.
    Save changes back to the original CSV files in base_dir and save processed prediction data.
    Only the last three rows for each player are kept in the prediction files.
    """
    positions_mapping = {
        'GK': 'gk.csv',
        'DEF': 'def.csv',
        'MID': 'mid.csv',
        'FWD': 'fwd.csv'
    }

    engineer_features_positions = {
        'GK': 'goalkeepers',
        'DEF': 'defenders',
        'MID': 'midfielders',
        'FWD': 'forwards'
    }

    season_dir = os.path.join(base_dir, season)
    if not os.path.exists(season_dir):
        print(f"Season directory {season_dir} does not exist.")
        return

    prediction_season_dir = os.path.join(prediction_dir, season, gw_folder_name)
    os.makedirs(prediction_season_dir, exist_ok=True)

    position_dfs = {position: pd.DataFrame() for position in positions_mapping.keys()}

    for position, output_file in positions_mapping.items():
        position_dir = os.path.join(season_dir, position)

        if not os.path.exists(position_dir):
            print(f"Position directory {position_dir} does not exist. Skipping.")
            continue

        for player_file in os.listdir(position_dir):
            player_file_path = os.path.join(position_dir, player_file)
            try:
                df = pd.read_csv(player_file_path)
            except pd.errors.EmptyDataError:
                print(f"File {player_file_path} is empty or corrupted. Skipping.")
                continue

            # Handle edge cases with minimum rows for calculations
            if len(df) >= 3:
                df.iloc[-2, df.columns.get_loc('name')] = df.iloc[-3]['name']
                df.iloc[-1, df.columns.get_loc('name')] = df.iloc[-3]['name']

            # Pre-calculate specific stats for MID and FWD positions
            if position in ['MID', 'FWD']:
                if 'last_season_goals' in df.columns and 'last_season_xG' in df.columns:
                    df['goal_stat'] = df['last_season_goals'] + df['last_season_xG']
                if 'last_season_assists' in df.columns and 'last_season_xA' in df.columns:
                    df['assist_stat'] = df['last_season_assists'] + df['last_season_xA']

            # Apply feature engineering to the dataframe
            try:
                df = engineer_features(df, engineer_features_positions[position])
                df.to_csv(player_file_path, index=False)
                print(f"Updated original file: {player_file_path} with feature engineering changes.")
            except Exception as e:
                print(f"Error applying feature engineering for {player_file_path}: {e}")
                continue

            # Append processed player data to the combined dataframe for this position
            position_dfs[position] = pd.concat([position_dfs[position], df])

        # After processing all player files for this position
        if not position_dfs[position].empty:
            # Apply feature engineering to the combined data
            try:
                position_dfs[position] = engineer_features(position_dfs[position], engineer_features_positions[position])
            except Exception as e:
                print(f"Error applying feature engineering for combined {position} data: {e}")
                continue

            # Keep only the last three rows for each player
            try:
                position_dfs[position] = position_dfs[position].groupby('name').tail(3).reset_index(drop=True)
            except Exception as e:
                print(f"Error filtering last three rows for {position}: {e}")
                continue

        # Save processed data for this position
        output_path = os.path.join(prediction_season_dir, output_file)
        position_dfs[position] = position_dfs[position].round(2)
        position_dfs[position].to_csv(output_path, index=False)
        print(f"{output_file} saved in {prediction_season_dir} with {len(position_dfs[position])} rows.")

    print("Prediction data processing completed.")


def main_create_prediction(gw_folder_name, seasons=['2024-25']):
    """Main method for creating prediction data with GW folder name input."""
    CURRENT_SEASON = seasons[0]; 
    GW_DIR = f'prediction_data/{CURRENT_SEASON}/'
    PRICES_FILE = 'current_prices.csv'

    create_prediction_data('player_data', 'prediction_data', gw_folder_name, CURRENT_SEASON)

    add_price_to_prediction_data(GW_DIR + gw_folder_name, PRICES_FILE)
    predict_gw(gw_folder_name)
    
    # Example usage
    
    '''
    merge_player_weeks(f'{GW_DIR}{gw_folder_name}/def.csv', f'{GW_DIR}{gw_folder_name}/DEF.csv')
    merge_player_weeks(f'{GW_DIR}{gw_folder_name}/gk.csv', f'{GW_DIR}{gw_folder_name}/GK.csv')
    merge_player_weeks(f'{GW_DIR}{gw_folder_name}/mid.csv', f'{GW_DIR}{gw_folder_name}/MID.csv')
    merge_player_weeks(f'{GW_DIR}{gw_folder_name}/fwd.csv', f'{GW_DIR}{gw_folder_name}/FWD.csv')

    combine_csv_files(gw_folder_name)
    

    '''