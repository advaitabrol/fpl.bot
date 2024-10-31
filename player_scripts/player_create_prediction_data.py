import os
import numpy as np
import pandas as pd

def create_prediction_data(base_dir, prediction_dir, gw_folder_name):
    positions_mapping = {
        'GK': 'gk.csv',
        'DEF': 'def.csv',
        'MID': 'mid.csv',
        'FWD': 'fwd.csv'
    }
    season_dir = os.path.join(base_dir, '2024-25')
    
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

            feature_sets = {
                'GK': ['clean_sheets', 'saves', 'goals_conceded', 'expected_goals_conceded', 'bonus'],
                'DEF': ['clean_sheets', 'goals', 'xG', 'goals_conceded', 'expected_goals_conceded', 'ict_index', 'bonus'],
                'MID': ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus', 'key_passes', 'shots'],
                'FWD': ['goals', 'assists', 'form', 'xG', 'xA', 'ict_index', 'bonus', 'shots', 'key_passes']
            }
            for feature in feature_sets[position]:
                if feature in calc_df.columns:
                    calc_df[feature] = calc_df[feature].fillna(0)
                    calc_df[f'{feature}_3g_avg'] = calc_df.groupby('name')[feature].transform(lambda x: x.rolling(3, min_periods=1).mean())
                    calc_df[f'{feature}_5g_avg'] = calc_df.groupby('name')[feature].transform(lambda x: x.rolling(5, min_periods=1).mean())

            for feature in ['next_week_specific_fixture_difficulty', 'next_week_holistic_fixture_difficulty']:
                if feature in df.columns:
                    df[feature] = df[feature].fillna(0)
                    df[f'{feature}_3g_avg'] = df.groupby('name')[feature].transform(lambda x: x.rolling(3, min_periods=1).mean())
                    df[f'{feature}_5g_avg'] = df.groupby('name')[feature].transform(lambda x: x.rolling(5, min_periods=1).mean())

            per_90_features = {
                'GK': ['saves'],
                'DEF': ['clean_sheets', 'goals', 'xG', 'assists', 'ict_index'],
                'MID': ['goals', 'assists', 'key_passes', 'shots', 'ict_index'],
                'FWD': ['goals', 'assists', 'ict_index']
            }
            calc_df['cumulative_minutes'] = calc_df.groupby('name')['minutes'].cumsum()
            for feature in per_90_features[position]:
                if feature in calc_df.columns:
                    calc_df[f'cumulative_{feature}'] = calc_df.groupby('name')[feature].cumsum()
                    calc_df[f'{feature}_per_90'] = np.where(calc_df['cumulative_minutes'] > 0,
                                                            calc_df[f'cumulative_{feature}'] / (calc_df['cumulative_minutes'] / 90), 0)

            calc_df.drop(columns=['cumulative_minutes'] + [f'cumulative_{feature}' for feature in per_90_features[position] if f'cumulative_{feature}' in calc_df.columns], inplace=True)
            
            interaction_terms = {
                'DEF': [('clean_sheets', 'goals'), ('ict_index', 'clean_sheets'), ('xG', 'expected_goals_conceded')],
                'MID': [('key_passes', 'assists'), ('shots', 'goals')],
                'FWD': [('goals', 'ict_index')],
                'GK': [('saves', 'clean_sheets'), ('expected_goals_conceded','saves')]
            }
            for (feat1, feat2) in interaction_terms[position]:
                if f'{feat1}_3g_avg' in calc_df.columns and f'{feat2}_3g_avg' in calc_df.columns:
                    calc_df[f'{feat1}_{feat2}_3g'] = calc_df[f'{feat1}_3g_avg'] * calc_df[f'{feat2}_3g_avg']


                if f'{feat1}_5g_avg' in calc_df.columns and f'{feat2}_5g_avg' in calc_df.columns:
                    calc_df[f'{feat1}_{feat2}_5g'] = calc_df[f'{feat1}_5g_avg'] * calc_df[f'{feat2}_5g_avg']


            for col in calc_df.columns:
                if col.endswith('_3g_avg') or col.endswith('_5g_avg') or col.endswith('_per_90') or col.endswith('_3g') or col.endswith('_5g'):
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

# Example usage with gw_folder_name as an argument
create_prediction_data('player_data', 'prediction_data', 'GW10-12')

