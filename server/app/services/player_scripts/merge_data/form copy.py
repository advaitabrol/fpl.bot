import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from fuzzywuzzy import process

player_position_changes = {
    "Kai_Havertz": "FWD"
}

def calculate_form(series, num_games=5):
    return series.rolling(window=num_games, min_periods=1).mean()

def get_last_season_stats(player_name, last_season_dir, position):
    # Check for position changes
    old_position = player_position_changes.get(player_name, position)

    # Match player file using fuzzy matching
    position_dir = os.path.join(last_season_dir, old_position)
    if not os.path.exists(position_dir):
        print(f"Directory {position_dir} does not exist. Skipping.")
        return {}

    player_files = os.listdir(position_dir)
    matched_file = process.extractOne(player_name, player_files, score_cutoff=75)

    last_season_file = ''
    if matched_file: 
        last_season_file = os.path.join(position_dir, matched_file[0])
    

    stats = {}
    last_season_df = pd.DataFrame(); 
    if last_season_file and os.path.exists(last_season_file):
        last_season_df = pd.read_csv(last_season_file)

    if position in ['FWD', 'MID']:
        stats = {
            'last_season_goals': last_season_df['goals'].sum() if 'goals' in last_season_df.columns else 0,
            'last_season_assists': last_season_df['assists'].sum() if 'assists' in last_season_df.columns else 0,
            'last_season_xG': last_season_df['xG'].sum() if 'xG' in last_season_df.columns else 0,
            'last_season_xA': last_season_df['xA'].sum() if 'xA' in last_season_df.columns else 0,
            'last_season_points_per_minute': last_season_df['total_points'].sum() / last_season_df['minutes'].sum() if 'total_points' in last_season_df.columns and last_season_df['minutes'].sum() > 0 else 0                
            }
    elif position == 'DEF':
        clean_sheet_games = last_season_df[(last_season_df['minutes'] > 60) & (last_season_df['clean_sheets'] == 1.0)] if 'clean_sheets' in last_season_df.columns else pd.DataFrame()
        total_games = last_season_df[last_season_df['minutes'] > 60] if 'minutes' in last_season_df.columns else pd.DataFrame()
        clean_sheet_prob = len(clean_sheet_games) / len(total_games) if not total_games.empty else 0
        stats = {
            'last_season_xG': last_season_df['xG'].sum() if 'xG' in last_season_df.columns else 0,
            'last_season_xA': last_season_df['xA'].sum() if 'xA' in last_season_df.columns else 0,
            'last_season_clean_sheet_probability': clean_sheet_prob,
            'last_season_expected_goals_conceded': last_season_df['expected_goals_conceded'].sum() if 'expected_goals_conceded' in last_season_df.columns else 0
        }
    elif position == 'GK':
        clean_sheet_games = last_season_df[(last_season_df['minutes'] > 60) & (last_season_df['clean_sheets'] == 1.0)] if 'clean_sheets' in last_season_df.columns else pd.DataFrame()
        total_games = last_season_df[last_season_df['minutes'] > 60] if 'minutes' in last_season_df.columns else pd.DataFrame()
        clean_sheet_prob = len(clean_sheet_games) / len(total_games) if not total_games.empty else 0
        stats = {
            'last_season_clean_sheet_probability': clean_sheet_prob,
            'last_season_total_saves': last_season_df['saves'].sum() if 'saves' in last_season_df.columns else 0,
            'last_season_expected_goals_conceded': last_season_df['expected_goals_conceded'].sum() if 'expected_goals_conceded' in last_season_df.columns else 0
        }
    return stats

def process_player_file(player_path, last_season_dir, position, season):
    try:
        df = pd.read_csv(player_path)
    except pd.errors.EmptyDataError:
        print(f"File {player_path} is empty or corrupted. Skipping.")
        return

    # Check for required columns
    if not all(col in df.columns for col in ['total_points', 'minutes']):
        print(f"File {player_path} is missing required columns. Skipping.")
        return

    # Calculate form columns
    df['form'] = calculate_form(df['total_points'])
    df['xG&A_form'] = calculate_form(df[['xG', 'assists']].sum(axis=1)) if 'xG' in df.columns and 'assists' in df.columns else 0
    df['minutes_per_game'] = df['minutes'].cumsum() / (df.index + 1)

    # Calculate clean sheet probability progressively for defenders and goalkeepers
    if position in ['DEF', 'GK'] and 'clean_sheets' in df.columns:
        df['clean_sheet_probability'] = df.apply(
            lambda row: len(df.loc[:row.name][(df.loc[:row.name, 'minutes'] > 60) & (df.loc[:row.name, 'clean_sheets'] == 1)]) / 
            len(df.loc[:row.name][df.loc[:row.name, 'minutes'] > 0]) if len(df.loc[:row.name][df.loc[:row.name, 'minutes'] > 0]) > 0 else 0,
            axis=1
        )

    # Calculate saves per game progressively for goalkeepers (only if minutes > 90)
    if position == 'GK' and 'saves' in df.columns:
        df['saves_per_game'] = df.apply(
            lambda row: df.loc[:row.name, 'saves'][df.loc[:row.name, 'minutes'] > 0].mean() if len(df.loc[:row.name, 'saves'][df.loc[:row.name, 'minutes'] > 0]) > 0 else 0,
            axis=1
        )

    # Retrieve last season stats for specific seasons
    if season in ['2022-23', '2023-24', '2024-25']:
        player_name = os.path.splitext(os.path.basename(player_path))[0]
        last_season_stats = get_last_season_stats(player_name, last_season_dir, position)

        # Update only relevant columns based on position
        for stat, value in last_season_stats.items():
            df[stat] = value

    # Round numerical columns to two decimal places
    df = df.round(2)

    # Save the updated dataframe back to the CSV file
    df.to_csv(player_path, index=False)
    #print(f"Processed and saved {player_path}")

def process_player_data(root_dir='player_data', seasons=['2022-23', '2023-24', '2024-25']):
    positions = ['FWD', 'MID', 'DEF', 'GK']

    with ThreadPoolExecutor() as executor:
        for season in seasons:
            last_season_dir = os.path.join(root_dir, f'{int(season[:4])-1}-{int(season[5:7])-1}')
            for position in positions:
                position_dir = os.path.join(root_dir, season, position)
                if not os.path.exists(position_dir):
                    print(f"Directory {position_dir} does not exist. Skipping.")
                    continue

                player_files = [os.path.join(position_dir, player_file) for player_file in os.listdir(position_dir) if player_file.endswith('.csv')]
                futures = [executor.submit(process_player_file, player_path, last_season_dir, position, season) for player_path in player_files]

                # Optionally, collect results to ensure all tasks complete
                for future in futures:
                    future.result()

# Example usage
if __name__ == "__main__":
    process_player_data()