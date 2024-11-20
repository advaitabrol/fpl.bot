import os
import numpy as np
import pandas as pd
import argparse

from prediction_data.add_prices import add_price_to_prediction_data;
from prediction_data.predict_gw import predict_gw; 
from prediction_data.reformat import merge_player_weeks;
from prediction_data.consolidate import combine_csv_files; 

def create_prediction_data(base_dir, prediction_dir, gw_folder_name, season='2024-25'):
    positions_mapping = {
        'GK': 'gk.csv',
        'DEF': 'def.csv',
        'MID': 'mid.csv',
        'FWD': 'fwd.csv'
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

            if position in ['MID', 'FWD']:
                if 'last_season_goals' in df.columns and 'last_season_xG' in df.columns:
                    df['goal_stat'] = df['last_season_goals'] + df['last_season_xG']
                if 'last_season_assists' in df.columns and 'last_season_xA' in df.columns:
                    df['assist_stat'] = df['last_season_assists'] + df['last_season_xA']

            calc_df = df.iloc[:-2].copy() if len(df) > 2 else df.copy()

            ##ROLLING AVERAGES 
            rolling_average_sets = {
                'GK': ['clean_sheets', 'saves', 'goals_conceded', 'expected_goals_conceded', 'bonus'],
                'DEF': ['clean_sheets', 'form', 'xG', 'xA', 'expected_goals_conceded', 'ict_index', 'bonus'],
                'MID': ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus', 'key_passes', 'shots'],
                'FWD': ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus', 'shots', 'key_passes']
            }
            for feature in rolling_average_sets[position]:
                if feature in calc_df.columns:
                    calc_df[feature] = calc_df[feature].fillna(0)
                    calc_df[f'{feature}_3g_avg'] = calc_df.groupby('name')[feature].transform(lambda x: x.rolling(3, min_periods=1).mean())
                    calc_df[f'{feature}_5g_avg'] = calc_df.groupby('name')[feature].transform(lambda x: x.rolling(5, min_periods=1).mean())

            ##WEIGHTED AVERAGES
            weighted_average_sets = {
                'GK': ['clean_sheets', 'saves', 'expected_goals_conceded'],
                'DEF': ['clean_sheets', 'form', 'xG', 'xA', 'ict_index', 'expected_goals_conceded', 'bonus'],
                'MID': ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus'],
                'FWD': ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus']
            }
            for feature in weighted_average_sets[position]:
                if feature in calc_df.columns:
                    calc_df[feature] = calc_df[feature].fillna(0)
                    calc_df[feature] = calc_df[feature].astype(float) 
                    calc_df[f'{feature}_ema'] = calc_df.groupby('name')[feature].transform(lambda x: x.ewm(span=5, adjust=False).mean())

            ##FIXTURE FEATURES --> STRAIGHT INTO THE DATAFRAME RATHER THAN CALC
            for feature in ['next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty']:
                if feature in df.columns:
                    df[feature] = df[feature].fillna(0)
                    df[f'{feature}_3g_avg'] = df.groupby('name')[feature].transform(lambda x: x.rolling(3, min_periods=1).mean())
                    df[f'{feature}_5g_avg'] = df.groupby('name')[feature].transform(lambda x: x.rolling(5, min_periods=1).mean())

            #MOMENTUM FEATURES
            momentum_sets = {
                'MID': ['form', 'xG', 'xA'],
                'FWD': ['form', 'xG', 'xA'],
                'DEF': ['expected_goals_conceded', 'form'], 
                'GK': ['expected_goals_conceded', 'saves'],
            } 
            for feature in momentum_sets[position]:
                if feature in calc_df.columns:
                 # Calculate momentum as the difference from a 3-game average
                    calc_df[f'{feature}_momentum_3g'] = calc_df[feature] - calc_df[f'{feature}_3g_avg']
                    # Calculate momentum as the difference from a 5-game average
                    calc_df[f'{feature}_momentum_5g'] = calc_df[feature] - calc_df[f'{feature}_5g_avg']

            ##PER90 FEATURES
            per_90_features = {
                'GK': ['saves'],
                'DEF': ['clean_sheets',  'xA', 'xG', 'ict_index'],
                'MID': ['goals', 'assists', 'xA', 'xG', 'key_passes', 'shots', 'ict_index'],
                'FWD': ['goals', 'assists', 'xA', 'xG', 'ict_index']
            }
            calc_df['cumulative_minutes'] = calc_df.groupby('name')['minutes'].cumsum()
            for feature in per_90_features[position]:
                if feature in calc_df.columns:
                    calc_df[f'cumulative_{feature}'] = calc_df.groupby('name')[feature].cumsum()
                    calc_df[f'{feature}_per_90'] = np.where(calc_df['cumulative_minutes'] > 0,
                                                            calc_df[f'cumulative_{feature}'] / (calc_df['cumulative_minutes'] / 90), 0)

            calc_df.drop(columns=['cumulative_minutes'] + [f'cumulative_{feature}' for feature in per_90_features[position] if f'cumulative_{feature}' in calc_df.columns], inplace=True)
            
            interaction_terms = {
                'DEF': [('clean_sheets', 'expected_goals_conceded'), ('form', 'next_week_specific_fixture_difficulty')],
                'MID': [('key_passes', 'assists'), ('shots', 'goals'), ('form', 'next_week_specific_fixture_difficulty')],
                'FWD': [('goals', 'ict_index'), ('form', 'next_week_specific_fixture_difficulty')],
                'GK': [('saves', 'clean_sheets'), ('expected_goals_conceded','saves')]
            }
            for (feat1, feat2) in interaction_terms[position]:
                if(feat2 == 'next_week_specific_fixture_difficulty'): 
                    if 'form_3g_avg' in calc_df.columns and 'next_week_specific_fixture_difficulty' in calc_df.columns: 
                            calc_df['form_difficulty_3g'] = calc_df['form_3g_avg'] * calc_df['next_week_specific_fixture_difficulty']
                else: 
                    if f'{feat1}_3g_avg' in calc_df.columns and f'{feat2}_3g_avg' in calc_df.columns:
                        calc_df[f'{feat1}_{feat2}_3g'] = calc_df[f'{feat1}_3g_avg'] * calc_df[f'{feat2}_3g_avg']

                    if f'{feat1}_5g_avg' in calc_df.columns and f'{feat2}_5g_avg' in calc_df.columns:
                        calc_df[f'{feat1}_{feat2}_5g'] = calc_df[f'{feat1}_5g_avg'] * calc_df[f'{feat2}_5g_avg']


            for col in calc_df.columns:
                if col.endswith('_3g_avg') or col.endswith('_5g_avg') or col.endswith('_per_90') or col.endswith('_3g') or col.endswith('_5g') or col.endswith('_ema'):
                    df.loc[df.index[:-2], col] = calc_df[col]

            if len(df) >= 3:
                last_three_rows = df.iloc[-3:].copy()
                exclude_cols = [f'next_week_{difficulty}_3g_avg' for difficulty in ['specific_fixture_difficulty', 'holistic_fixture_difficulty']] + \
                               [f'next_week_{difficulty}_5g_avg' for difficulty in ['specific_fixture_difficulty', 'holistic_fixture_difficulty']]
                
                for col in last_three_rows.columns:
                    if col == 'name':
                        last_three_rows.iloc[-2, last_three_rows.columns.get_loc(col)] = last_three_rows.iloc[-3][col]
                        last_three_rows.iloc[-1, last_three_rows.columns.get_loc(col)] = last_three_rows.iloc[-3][col]
                    elif col.endswith('_3g_avg') or col.endswith('_5g_avg') or col.endswith('_per_90'):
                        if col not in exclude_cols:
                            if pd.isna(last_three_rows.iloc[-2][col]):
                                last_three_rows.iloc[-2, last_three_rows.columns.get_loc(col)] = last_three_rows.iloc[-3][col]
                            if pd.isna(last_three_rows.iloc[-1][col]):
                                last_three_rows.iloc[-1, last_three_rows.columns.get_loc(col)] = last_three_rows.iloc[-3][col]
                    elif last_three_rows[col].dtype in [np.float64, np.int64]:
                        if pd.isna(last_three_rows.iloc[-2][col]):
                            last_three_rows.iloc[-2, last_three_rows.columns.get_loc(col)] = last_three_rows.iloc[-3][col]
                        if pd.isna(last_three_rows.iloc[-1][col]):
                            last_three_rows.iloc[-1, last_three_rows.columns.get_loc(col)] = last_three_rows.iloc[-3][col]

                position_dfs[position] = pd.concat([position_dfs[position], last_three_rows])

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
    PRICES_FILE = 'player_scripts/current_prices.csv'

    create_prediction_data('player_data', 'prediction_data', gw_folder_name, CURRENT_SEASON)

    add_price_to_prediction_data(GW_DIR + gw_folder_name, PRICES_FILE)
    predict_gw(gw_folder_name)
    
    # Example usage
    merge_player_weeks(f'prediction_data/{CURRENT_SEASON}/{gw_folder_name}/def.csv', f'prediction_data/{CURRENT_SEASON}/{gw_folder_name}/DEF.csv')
    merge_player_weeks(f'prediction_data/{CURRENT_SEASON}/{gw_folder_name}/gk.csv', f'prediction_data/{CURRENT_SEASON}/{gw_folder_name}/GK.csv')
    merge_player_weeks(f'prediction_data/{CURRENT_SEASON}/{gw_folder_name}/mid.csv', f'prediction_data/{CURRENT_SEASON}/{gw_folder_name}/MID.csv')
    merge_player_weeks(f'prediction_data/{CURRENT_SEASON}/{gw_folder_name}/fwd.csv', f'prediction_data/{CURRENT_SEASON}/{gw_folder_name}/FWD.csv')

    combine_csv_files(gw_folder_name)
    

