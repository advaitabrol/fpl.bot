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
    positions_mapping = {
        'GK': 'gk.csv',
        'DEF': 'def.csv',
        'MID': 'mid.csv',
        'FWD': 'fwd.csv'
    }

    # Mapping shorthand to full position names for engineer_features
    position_name_mapping = {
        'GK': 'goalkeepers',
        'DEF': 'defenders',
        'MID': 'midfielders',
        'FWD': 'forwards'
    }

    season_dir = os.path.join(base_dir, season)
    
    if not os.path.exists(season_dir):
        print(f"Season directory {season_dir} does not exist.")
        return
    
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

            if len(df) >= 3:
                df.iloc[-2, df.columns.get_loc('name')] = df.iloc[-3]['name']
                df.iloc[-1, df.columns.get_loc('name')] = df.iloc[-3]['name']

            # Create a copy of the relevant data for feature engineering
            calc_df = df.iloc[:-2].copy() if len(df) > 2 else df.copy()

            # Map shorthand to full position name and call engineer_features
            full_position_name = position_name_mapping.get(position)
            if full_position_name:
                try:
                    calc_df = engineer_features(calc_df, full_position_name)
                except Exception as e:
                    print(f"Error applying feature engineering for {player_file_path}: {e}")
                    continue
            else:
                print(f"Position {position} does not have a corresponding full name. Skipping.")
                continue

            # Update the original dataframe with the new engineered features
            for col in calc_df.columns:
                if col not in df.columns:  # Add new columns to df if they don't exist
                    df[col] = np.nan
                if col.endswith('_3g_avg') or col.endswith('_5g_avg') or col.endswith('_per_90') or col.endswith('_3g') or col.endswith('_5g') or col.endswith('_ema'):
                    df.loc[df.index[:-2], col] = calc_df[col]
                else:
                    # For columns not matching the suffix, still update excluding the last two
                    df.loc[df.index[:-2], col] = calc_df[col]

            # Keep only the last three rows for prediction data
            if len(df) >= 3:
                last_three_rows = df.iloc[-3:].copy()
                exclude_cols = (
                    [f'next_week_{difficulty}_3g_avg' for difficulty in ['specific_fixture_difficulty', 'holistic_fixture_difficulty']] +
                    [f'next_week_{difficulty}_5g_avg' for difficulty in ['specific_fixture_difficulty', 'holistic_fixture_difficulty']] +
                    [
                        'name',
                        'team',
                        'opponent_team',
                        'date',
                        'was_home',
                        'next_week_points',
                        'next_team',
                        'next_was_home',
                        'next_opponent_team',
                        'next_week_specific_fixture_difficulty',
                        'next_week_holistic_fixture_difficulty',
                        'next_week_specific_fixture_difficulty_ema',
                        'next_week_specific_fixture_difficulty_momentum_3g',
                        'next_week_specific_fixture_difficulty_momentum_5g',
                        'next_week_holistic_fixture_difficulty_ema',
                        'next_week_holistic_fixture_difficulty_momentum_3g',
                        'next_week_holistic_fixture_difficulty_momentum_5g',
                    ]
                )

                for col in last_three_rows.columns:
                    if col == 'name':
                        last_three_rows.iloc[-2, last_three_rows.columns.get_loc(col)] = last_three_rows.iloc[-3][col]
                        last_three_rows.iloc[-1, last_three_rows.columns.get_loc(col)] = last_three_rows.iloc[-3][col]
                    elif col not in exclude_cols:
                        last_three_rows.iloc[-2, last_three_rows.columns.get_loc(col)] = last_three_rows.iloc[-3][col]
                        last_three_rows.iloc[-1, last_three_rows.columns.get_loc(col)] = last_three_rows.iloc[-3][col]

                # Calculate next_week difficulty stats for the last two rows
                for difficulty in ['specific_fixture_difficulty', 'holistic_fixture_difficulty']:
                    col_3g_avg = f'next_week_{difficulty}_3g_avg'
                    col_5g_avg = f'next_week_{difficulty}_5g_avg'
                    col_ema = f'next_week_{difficulty}_ema'
                    col_momentum_3g = f'next_week_{difficulty}_momentum_3g'
                    col_momentum_5g = f'next_week_{difficulty}_momentum_5g'

                    # Check if required columns exist in the data
                    if all(col in last_three_rows.columns for col in [col_3g_avg, col_5g_avg, col_ema, col_momentum_3g, col_momentum_5g]):
                        for idx in [-2, -1]:  # Only update for the last two rows
                            base_value = last_three_rows.iloc[idx][f'next_week_{difficulty}']

                            # Determine the previous row based on the current row's index
                            if idx == -2:
                                # For second-to-last row, use the row before it
                                previous_row = last_three_rows.iloc[idx - 1] if idx > -len(last_three_rows) else None
                                divisor_3g, divisor_5g = 4, 6
                            else:
                                # For last row, use the second-to-last row
                                previous_row = last_three_rows.iloc[-2]
                                divisor_3g, divisor_5g = 5, 7

                            if pd.notna(base_value):  # Only calculate if the base value exists
                                # Include the previous row's value if it exists
                                additional_value = previous_row[f'next_week_{difficulty}'] if previous_row is not None and pd.notna(previous_row[f'next_week_{difficulty}']) else 0

                                # Calculate 3g_avg and 5g_avg
                                last_three_rows.iloc[idx, last_three_rows.columns.get_loc(col_3g_avg)] = (base_value + additional_value) / divisor_3g
                                last_three_rows.iloc[idx, last_three_rows.columns.get_loc(col_5g_avg)] = (base_value + additional_value) / divisor_5g

                                # Calculate ema (using exponential smoothing similar to engineer_features)
                                smoothing_factor = 2 / (divisor_3g + 1)  # Slightly adjusted for 3g divisor
                                previous_ema = previous_row[col_ema] if previous_row is not None and pd.notna(previous_row[col_ema]) else 0
                                ema_value = (base_value * smoothing_factor) + (previous_ema * (1 - smoothing_factor))
                                last_three_rows.iloc[idx, last_three_rows.columns.get_loc(col_ema)] = ema_value

                                # Calculate momentum (difference from averages)
                                last_three_rows.iloc[idx, last_three_rows.columns.get_loc(col_momentum_3g)] = base_value - last_three_rows.iloc[idx, last_three_rows.columns.get_loc(col_3g_avg)]
                                last_three_rows.iloc[idx, last_three_rows.columns.get_loc(col_momentum_5g)] = base_value - last_three_rows.iloc[idx, last_three_rows.columns.get_loc(col_5g_avg)]


                                # Append the updated rows back to the position DataFrame
                position_dfs[position] = pd.concat([position_dfs[position], last_three_rows])

    # Save processed prediction data
    prediction_season_dir = os.path.join(prediction_dir, '2024-25', gw_folder_name)
    os.makedirs(prediction_season_dir, exist_ok=True)

    for position, output_file in positions_mapping.items():
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
    
    #Example usage
    
    
    merge_player_weeks(f'{GW_DIR}{gw_folder_name}/def.csv', f'{GW_DIR}{gw_folder_name}/DEF.csv')
    merge_player_weeks(f'{GW_DIR}{gw_folder_name}/gk.csv', f'{GW_DIR}{gw_folder_name}/GK.csv')
    merge_player_weeks(f'{GW_DIR}{gw_folder_name}/mid.csv', f'{GW_DIR}{gw_folder_name}/MID.csv')
    merge_player_weeks(f'{GW_DIR}{gw_folder_name}/fwd.csv', f'{GW_DIR}{gw_folder_name}/FWD.csv')

    combine_csv_files(gw_folder_name)
    

    