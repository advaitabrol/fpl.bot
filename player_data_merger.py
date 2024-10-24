import os
import pandas as pd
import shutil
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


###MERGE THE FPL DATA WITH THE UNDERSTAT DATA 
def merge_fpl_and_understat_data(seasons=['2021-22', '2022-23', '2023-24', '2024-25'], merged_data_dir = 'player_data'):
    positions = ['FWD', 'MID', 'DEF', 'GK']
    
    # Create the merged_data directory with subdirectories for each season and position
    for season in seasons:
        season_dir = os.path.join(merged_data_dir, season)
        os.makedirs(season_dir, exist_ok=True)
        
        for position in positions:
            position_dir = os.path.join(season_dir, position)
            os.makedirs(position_dir, exist_ok=True)

    # Step 2: Define paths to fpl_gw_data and understat_data directories
    fpl_gw_data_dir = 'fpl_gw_data'
    understat_data_dir = 'understat_data'
    
    # Function to find the best match using fuzzy matching
    def find_best_match(player_name, understat_files):
        best_match = process.extractOne(player_name, understat_files, scorer=fuzz.partial_ratio)
        if best_match and best_match[1] >= 65:  # Adjust the threshold as needed
            return best_match[0]
        return None

    # Step 3: Process each CSV file in fpl_gw_data
    for season in seasons:
        fpl_season_dir = os.path.join(fpl_gw_data_dir, season)
        understat_season_dir = os.path.join(understat_data_dir, season)
        
        if os.path.isdir(fpl_season_dir) and os.path.isdir(understat_season_dir):
            for fpl_csv in os.listdir(fpl_season_dir):
                if fpl_csv.endswith('.csv'):
                    fpl_csv_path = os.path.join(fpl_season_dir, fpl_csv)
                    
                    # Read the fpl_gw_data CSV
                    fpl_df = pd.read_csv(fpl_csv_path)
                    
                    # Get player name and position
                    player_name = fpl_df['name'].iloc[0]  
                    player_position = fpl_df['position'].iloc[0]
                    
                    # Capitalize first and last name for fuzzy matching
                    player_name_capitalized = '_'.join([name.capitalize() for name in player_name.split()])
                    
                    # List of all Understat files in the corresponding season directory
                    understat_files = os.listdir(understat_season_dir)
                    
                    # Use fuzzy matching to find the best match for the player
                    best_match_file = find_best_match(player_name_capitalized, understat_files)
                    
                    if best_match_file:
                        understat_csv_path = os.path.join(understat_season_dir, best_match_file)
                        
                        # Read the understat CSV
                        understat_df = pd.read_csv(understat_csv_path)
                        
                        # Merge fpl_gw_data with understat_data based on matching kickoff_time and date
                        fpl_df['kickoff_time'] = pd.to_datetime(fpl_df['kickoff_time']).dt.date
                        understat_df['date'] = pd.to_datetime(understat_df['date']).dt.date
                        
                        # Perform a left join to keep all rows from fpl_df and fill missing data with NaN
                        merged_df = pd.merge(fpl_df, understat_df, left_on='kickoff_time', right_on='date', how='left')
                        
                        # Save the merged file in the correct position folder in merged_data
                        merged_output_dir = os.path.join(merged_data_dir, season, player_position)
                        merged_output_path = os.path.join(merged_output_dir, fpl_csv)
                        
                        merged_df.to_csv(merged_output_path, index=False)
                        print(f"Merged and saved {fpl_csv} to {merged_output_dir}")
                    else:
                        print(f"No matching Understat file found for {player_name}. Skipping {fpl_csv}.")
        else:
            print(f"Season folder {season} does not exist in both fpl_gw_data and understat_data.")


### CLEAN OUT ALL THE DATA POSITOIN BY POSITION
def process_fwd_data(merged_data_dir = 'player_data', seasons = ['2021-22', '2022-23', '2023-24', '2024-25']):
    # Define the path to the FWD folder in each season
    fwd_folder = 'FWD'

    # Columns to drop
    columns_to_drop = ['position_x', 'kickoff_time', 'time']

    # Column order as specified, including 'starts' as optional
    desired_column_order = [
        'name', 'team', 'opponent_team', 'date', 'was_home', 'position_y', 'minutes', 'goals', 'xG', 'assists', 'xA',
        'total_points', 'shots', 'key_passes', 'ict_index', 'bonus', 'starts'
    ]

    # Loop through each season and process the FWD folder
    for season in seasons:
        fwd_dir = os.path.join(merged_data_dir, season, fwd_folder)
        
        if os.path.isdir(fwd_dir):
            for csv_file in os.listdir(fwd_dir):
                if csv_file.endswith('.csv'):
                    file_path = os.path.join(fwd_dir, csv_file)
                    
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Step 1: Drop the unnecessary columns
                    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
                    
                    # Step 2: Convert 'starts' column to boolean (True/False) if it exists
                    if 'starts' in df.columns:
                        df['starts'] = df['starts'].apply(lambda x: True if x == 1 else False)
                    else:
                        print(f"'starts' column not found in {csv_file}, skipping the conversion.")
                    
                    # Step 3: Reorder the columns, but first check for missing columns
                    existing_columns = [col for col in desired_column_order if col in df.columns]
                    df = df[existing_columns]
                    
                    # Step 4: Rename 'position_y' to 'position' if 'position_y' exists
                    if 'position_y' in df.columns:
                        df = df.rename(columns={'position_y': 'position'})
                    
                    # Save the modified CSV file back
                    df.to_csv(file_path, index=False)
                    print(f"Processed and saved {csv_file} in {fwd_dir}")
        else:
            print(f"FWD directory for season {season} does not exist.")

def process_mid_data( merged_data_dir = 'player_data', seasons = ['2021-22', '2022-23', '2023-24', '2024-25']):
    # Define the path to the MID folder in each season
    mid_folder = 'MID'

    # Columns to drop
    columns_to_drop = ['position_x', 'kickoff_time', 'time']

    # Column order as specified for the MID folder
    desired_column_order = [
        'name', 'team', 'opponent_team', 'date', 'was_home', 'position_y', 'minutes', 'goals', 'xG', 'assists', 'xA',
        'total_points', 'shots', 'key_passes', 'ict_index', 'bonus', 'clean_sheets', 'goals_conceded', 'starts'
    ]

    # Loop through each season and process the MID folder
    for season in seasons:
        mid_dir = os.path.join(merged_data_dir, season, mid_folder)
        
        if os.path.isdir(mid_dir):
            for csv_file in os.listdir(mid_dir):
                if csv_file.endswith('.csv'):
                    file_path = os.path.join(mid_dir, csv_file)
                    
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Step 1: Drop the unnecessary columns
                    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
                    
                    # Step 2: Convert 'starts' column to boolean (True/False) if it exists
                    if 'starts' in df.columns:
                        df['starts'] = df['starts'].apply(lambda x: True if x == 1 else False)
                    else:
                        print(f"'starts' column not found in {csv_file}, skipping the conversion.")
                    
                    # Step 3: Reorder the columns, but first check for missing columns
                    existing_columns = [col for col in desired_column_order if col in df.columns]
                    df = df[existing_columns]
                    
                    # Step 4: Rename 'position_y' to 'position' if 'position_y' exists
                    if 'position_y' in df.columns:
                        df = df.rename(columns={'position_y': 'position'})
                    
                    # Save the modified CSV file back
                    df.to_csv(file_path, index=False)
                    print(f"Processed and saved {csv_file} in {mid_dir}")
        else:
            print(f"MID directory for season {season} does not exist.")

def process_def_data(merged_data_dir = 'player_data', seasons = ['2021-22', '2022-23', '2023-2024', '2024-25']):
    # Define the path to the DEF folder in each season
    def_folder = 'DEF'

    # Columns to drop
    columns_to_drop = ['position_x', 'kickoff_time', 'time', 'own_goals']

    # Column order as specified for the DEF folder
    desired_column_order = [
        'name', 'team', 'opponent_team', 'date', 'was_home', 'position_y', 'minutes', 'goals', 'xG', 'assists', 'xA',
        'total_points', 'shots', 'key_passes', 'ict_index', 'bonus', 'clean_sheets', 'goals_conceded', 'expected_goals_conceded', 'starts'
    ]

    # Loop through each season and process the DEF folder
    for season in seasons:
        def_dir = os.path.join(merged_data_dir, season, def_folder)
        
        if os.path.isdir(def_dir):
            for csv_file in os.listdir(def_dir):
                if csv_file.endswith('.csv'):
                    file_path = os.path.join(def_dir, csv_file)
                    
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Step 1: Drop the unnecessary columns
                    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
                    
                    # Step 2: Convert 'starts' column to boolean (True/False) if it exists
                    if 'starts' in df.columns:
                        df['starts'] = df['starts'].apply(lambda x: True if x == 1 else False)
                    else:
                        print(f"'starts' column not found in {csv_file}, skipping the conversion.")
                    
                    # Step 3: Reorder the columns, but first check for missing columns
                    existing_columns = [col for col in desired_column_order if col in df.columns]
                    df = df[existing_columns]
                    
                    # Step 4: Rename 'position_y' to 'position' if 'position_y' exists
                    if 'position_y' in df.columns:
                        df = df.rename(columns={'position_y': 'position'})
                    
                    # Save the modified CSV file back
                    df.to_csv(file_path, index=False)
                    print(f"Processed and saved {csv_file} in {def_dir}")
        else:
            print(f"DEF directory for season {season} does not exist.")

def process_gk_data(merged_data_dir = 'player_data', 
    seasons = ['2021-22', '2022-23', '2023-24', '2024-25']):
    # Define the path to the GK folder in each season
    gk_folder = 'GK'

    # Columns to drop
    columns_to_drop = ['position_x', 'kickoff_time', 'time', 'own_goals', 'goals', 'shots', 'xG', 'assists', 'key_passes', 'position_y']

    # Column order as specified for the GK folder
    desired_column_order = [
        'name', 'team', 'opponent_team', 'date', 'was_home', 'minutes', 'goals_conceded', 'expected_goals_conceded',
        'saves', 'penalties_saved', 'total_points', 'bonus', 'clean_sheets', 'xA', 'starts'
    ]

    # Loop through each season and process the GK folder
    for season in seasons:
        gk_dir = os.path.join(merged_data_dir, season, gk_folder)
        
        if os.path.isdir(gk_dir):
            for csv_file in os.listdir(gk_dir):
                if csv_file.endswith('.csv'):
                    file_path = os.path.join(gk_dir, csv_file)
                    
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Step 1: Drop the unnecessary columns
                    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
                    
                    # Step 2: Convert 'starts' and 'clean_sheets' columns to boolean (True/False) if they exist
                    if 'starts' in df.columns:
                        df['starts'] = df['starts'].apply(lambda x: True if x == 1 else False)
                    else:
                        print(f"'starts' column not found in {csv_file}, skipping the conversion.")
                    
                    if 'clean_sheets' in df.columns:
                        df['clean_sheets'] = df['clean_sheets'].apply(lambda x: True if x == 1 else False)
                    else:
                        print(f"'clean_sheets' column not found in {csv_file}, skipping the conversion.")
                    
                    # Step 3: Reorder the columns, but first check for missing columns
                    existing_columns = [col for col in desired_column_order if col in df.columns]
                    df = df[existing_columns]
                    
                    # Save the modified CSV file back
                    df.to_csv(file_path, index=False)
                    print(f"Processed and saved {csv_file} in {gk_dir}")
        else:
            print(f"GK directory for season {season} does not exist.")



###CALCULATE FORM STATS FOR FWDs and MIDs
def calculate_fwd_mid_form(series, num_games=5):
    form = []
    for i in range(len(series)):
        if i < num_games:
            form.append(series[:i].mean() if i > 0 else 0)  # Avoid empty slice error
        else:
            form.append(series[i-num_games:i].mean())
    return form
# Helper function to get last season's stats for a forward or midfielder
def get_fwd_mid_last_season_stats(player_name, last_season_dir, position):
    last_season_file = os.path.join(last_season_dir, position, player_name)
    
    print(f"Looking for {player_name} in {last_season_file}")  # Debugging
    
    if os.path.exists(last_season_file):
        last_season_df = pd.read_csv(last_season_file)
        if not last_season_df.empty:
            required_columns = ['goals', 'assists', 'xG', 'xA', 'total_points', 'minutes']
            if all(col in last_season_df.columns for col in required_columns):
                total_goals = last_season_df['goals'].sum()
                total_assists = last_season_df['assists'].sum()
                total_xG = last_season_df['xG'].sum()
                total_xA = last_season_df['xA'].sum()
                total_points = last_season_df['total_points'].sum()
                total_minutes = last_season_df['minutes'].sum()
                points_per_minute = total_points / total_minutes if total_minutes > 0 else 0
                
                print(f"Stats found for {player_name}: Goals: {total_goals}, Assists: {total_assists}, xG: {total_xG}, xA: {total_xA}, Points/Min: {points_per_minute}")  # Debugging
                
                return total_goals, total_assists, total_xG, total_xA, points_per_minute
            else:
                print(f"Missing columns in {last_season_file}")  # Debugging
    else:
        print(f"File {last_season_file} not found.")  # Debugging
    
    return 0, 0, 0, 0, 0
# Main function to process both forwards and midfielders data
def form_fwd_mid_data(root_dir='player_data', seasons=['2021-22', '2022-23', '2023-24', '2024-25'], positions=['FWD', 'MID']):
    for season in seasons:
        for position in positions:
            process_fwd_mid_position_data(season, position, root_dir)
# Function to process forward and midfielder position data for a season
def process_fwd_mid_position_data(season, position, root_dir):
    position_dir = os.path.join(root_dir, season, position)
    last_season_dir = os.path.join(root_dir, f'{int(season[:4])-1}-{int(season[5:7])-1}')
    
    if not os.path.exists(position_dir):
        print(f"Directory {position_dir} does not exist. Skipping.")
        return

    # Loop through each CSV file in the position directory
    for player_file in os.listdir(position_dir):
        player_path = os.path.join(position_dir, player_file)
        process_fwd_mid_player_file(player_path, season, last_season_dir, position)
# Function to process a single player's file (specific to forwards and midfielders)
def process_fwd_mid_player_file(player_path, season, last_season_dir, position):
    try:
        df = pd.read_csv(player_path)
    except pd.errors.EmptyDataError:
        print(f"File {player_path} is empty or corrupted. Skipping.")
        return
    
    required_columns = ['total_points', 'xG', 'assists', 'minutes']
    if not all(col in df.columns for col in required_columns):
        print(f"File {player_path} is missing required columns. Skipping.")
        return
    
    # 1. Create 'form' column
    df['form'] = calculate_fwd_mid_form(df['total_points'])
    
    # 2. Create 'xG&A_form' column
    df['xG&A_form'] = calculate_fwd_mid_form(df[['xG', 'assists']].sum(axis=1))
    
    # 3. Create 'minutes per game' column
    df['minutes_per_game'] = df['minutes'].cumsum() / (df.index + 1)
    
    # 4. Get last season stats for 2022-23 and 2023-24 seasons
    if season in ['2022-23', '2023-24', '2024-25']:
        player_file = os.path.basename(player_path)
        last_season_goals, last_season_assists, last_season_xG, last_season_xA, last_season_ppm = get_fwd_mid_last_season_stats(player_file, last_season_dir, position)
        df['last_season_goals'] = last_season_goals
        df['last_season_assists'] = last_season_assists
        df['last_season_xG'] = last_season_xG
        df['last_season_xA'] = last_season_xA
        df['last_season_points_per_minute'] = last_season_ppm
    
    # Round all numerical columns to two decimal places
    df = df.round(2)
    
    # Save the updated dataframe back to the CSV file
    df.to_csv(player_path, index=False)
    print(f"Processed and saved {player_path}")



###CALCULATE THE FORM FOR DEFENDERS
# Function to calculate form over last five games (or as many as available) for defenders
def calculate_defender_form(series, num_games=5):
    form = []
    for i in range(len(series)):
        if i < num_games:
            form.append(series[:i].mean() if i > 0 else 0)  # Avoid empty slice error
        else:
            form.append(series[i-num_games:i].mean())
    return form

# Function to get last season's stats for defenders
def get_last_season_stats_defender(player_name, last_season_dir):
    last_season_file = os.path.join(last_season_dir, 'DEF', player_name)
    
    print(f"Looking for {player_name} in {last_season_file}")  # Debugging
    
    if os.path.exists(last_season_file):
        last_season_df = pd.read_csv(last_season_file)
        if not last_season_df.empty:
            required_columns = ['xG', 'xA', 'expected_goals_conceded', 'minutes', 'clean_sheets']
            if all(col in last_season_df.columns for col in required_columns):
                total_xG = last_season_df['xG'].sum()
                total_xA = last_season_df['xA'].sum()
                total_expected_goals_conceded = last_season_df['expected_goals_conceded'].sum()
                
                clean_sheet_games = last_season_df[(last_season_df['minutes'] > 60) & (last_season_df['clean_sheets'] == True)]
                total_games = last_season_df[last_season_df['minutes'] > 60]
                if not total_games.empty:
                    last_season_clean_sheet_prob = len(clean_sheet_games) / len(total_games)
                else:
                    last_season_clean_sheet_prob = 0
                
                print(f"Stats found for {player_name}: xG: {total_xG}, xA: {total_xA}, Expected Goals Conceded: {total_expected_goals_conceded}, Clean Sheet Probability: {last_season_clean_sheet_prob}")  # Debugging
                
                return total_xG, total_xA, total_expected_goals_conceded, last_season_clean_sheet_prob
            else:
                print(f"Missing required columns in {last_season_file}")  # Debugging
    else:
        print(f"File {last_season_file} not found.")  # Debugging
    
    return 0, 0, 0, 0
# Function to calculate clean sheet probability for current season
def calculate_clean_sheet_probability(df):
    clean_sheet_games = df[(df['minutes'] > 60) & (df['clean_sheets'] == True)]
    total_games = df[df['minutes'] > 60]
    if not total_games.empty:
        return len(clean_sheet_games) / len(total_games)
    return 0
# Main function to process defender data across all seasons
def form_defender_data(root_dir='player_data', seasons=['2021-22', '2022-23', '2023-24', '2024-25']):
    for season in seasons:
        process_defender_season_data(season, root_dir)
# Function to process defender data for a specific season
def process_defender_season_data(season, root_dir):
    position_dir = os.path.join(root_dir, season, 'DEF')
    last_season_dir = os.path.join(root_dir, f'{int(season[:4])-1}-{int(season[5:7])-1}')
    
    if not os.path.exists(position_dir):
        print(f"Directory {position_dir} does not exist. Skipping.")
        return

    for player_file in os.listdir(position_dir):
        player_path = os.path.join(position_dir, player_file)
        process_defender_player_file(player_path, season, last_season_dir)
# Function to process a single defender's file
def process_defender_player_file(player_path, season, last_season_dir):
    try:
        df = pd.read_csv(player_path)
    except pd.errors.EmptyDataError:
        print(f"File {player_path} is empty or corrupted. Skipping.")
        return
    
    required_columns = ['total_points', 'xG', 'assists', 'minutes', 'clean_sheets']
    if not all(col in df.columns for col in required_columns):
        print(f"File {player_path} is missing required columns. Skipping.")
        return
    
    # 1. Create 'form' column if 'total_points' is available
    if 'total_points' in df.columns:
        df['form'] = calculate_defender_form(df['total_points'])
    
    # 2. Create 'xG&A_form' column if 'xG' and 'assists' are available
    if 'xG' in df.columns and 'assists' in df.columns:
        df['xG&A_form'] = calculate_defender_form(df[['xG', 'assists']].sum(axis=1))
    
    # 3. Create 'minutes per game' column if 'minutes' is available
    if 'minutes' in df.columns:
        df['minutes_per_game'] = df['minutes'].cumsum() / (df.index + 1)
    
    # 4. Calculate clean sheet probability if 'minutes' and 'clean_sheets' are available
    if 'minutes' in df.columns and 'clean_sheets' in df.columns:
        df['clean_sheet_probability'] = df['clean_sheets'].expanding().apply(lambda x: calculate_clean_sheet_probability(df[:x.index[-1]+1]), raw=False)
    
    # 5. Get last season stats (only for 2022-23 and 2023-24 seasons)
    if season in ['2022-23', '2023-24', '2024-25']:
        player_file = os.path.basename(player_path)
        last_season_xG, last_season_xA, last_season_expected_goals_conceded, last_season_clean_sheet_prob = get_last_season_stats_defender(player_file, last_season_dir)
        df['last_season_xG'] = last_season_xG
        df['last_season_xA'] = last_season_xA
        df['last_season_expected_goals_conceded'] = last_season_expected_goals_conceded
        df['last_season_clean_sheet_probability'] = last_season_clean_sheet_prob
    
    # Round all numerical columns to two decimal places
    df = df.round(2)
    
    # Save the updated dataframe back to the CSV file
    df.to_csv(player_path, index=False)
    print(f"Processed and saved {player_path}")



### CALCULATE THE FORM FOR GOALKEEPERS

# Helper function to calculate form over last five games (or as many as available)
def calculate_goalkeeper_form(series, num_games=5):
    form = []
    for i in range(len(series)):
        if i < num_games:
            form.append(series[:i].mean() if i > 0 else 0)  # Avoid empty slice error
        else:
            form.append(series[i-num_games:i].mean())
    return form
# Helper function to get last season stats for goalkeepers
def get_last_season_stats_goalkeeper(player_name, last_season_dir):
    last_season_file = os.path.join(last_season_dir, 'GK', player_name)
    
    print(f"Looking for {player_name} in {last_season_file}")  # Debugging
    
    if os.path.exists(last_season_file):
        last_season_df = pd.read_csv(last_season_file)
        if not last_season_df.empty:
            required_columns = ['penalties_saved', 'expected_goals_conceded', 'minutes', 'clean_sheets', 'saves']
            if all(col in last_season_df.columns for col in required_columns):
                total_penalties_saved = last_season_df['penalties_saved'].sum()
                total_expected_goals_conceded = last_season_df['expected_goals_conceded'].sum()
                total_saves = last_season_df['saves'].sum()
                
                # Calculate last season clean sheet probability
                clean_sheet_games = last_season_df[(last_season_df['minutes'] > 60) & (last_season_df['clean_sheets'] == True)]
                total_games = last_season_df[last_season_df['minutes'] > 60]
                if not total_games.empty:
                    last_season_clean_sheet_prob = len(clean_sheet_games) / len(total_games)
                else:
                    last_season_clean_sheet_prob = 0
                
                print(f"Stats found for {player_name}: Penalties Saved: {total_penalties_saved}, Expected Goals Conceded: {total_expected_goals_conceded}, Last Season Saves: {total_saves}, Last Season Clean Sheet Probability: {last_season_clean_sheet_prob}")  # Debugging
                
                return total_penalties_saved, total_expected_goals_conceded, last_season_clean_sheet_prob, total_saves
            else:
                print(f"Missing required columns in {last_season_file}: {set(required_columns) - set(last_season_df.columns)}")  # Debugging
    else:
        print(f"File {last_season_file} not found.")  # Debugging
    
    return 0, 0, 0, 0
# Helper function to calculate clean sheet probability for current season
def calculate_clean_sheet_probability(df):
    clean_sheet_games = df[(df['minutes'] > 60) & (df['clean_sheets'] == True)]
    total_games = df[df['minutes'] > 60]
    if not total_games.empty:
        return len(clean_sheet_games) / len(total_games)
    return 0
# Helper function to calculate saves per game
def calculate_saves_per_game(df):
    return df['saves'].cumsum() / (df.index + 1)
# Main function to process goalkeeper data across all seasons
def form_goalkeeper_data(root_dir='player_data', seasons=['2021-22', '2022-23', '2023-24', '2024-25']):
    for season in seasons:
        process_goalkeeper_season_data(season, root_dir)
# Function to process goalkeeper data for a specific season
def process_goalkeeper_season_data(season, root_dir):
    position_dir = os.path.join(root_dir, season, 'GK')
    last_season_dir = os.path.join(root_dir, f'{int(season[:4])-1}-{int(season[5:7])-1}')
    
    if not os.path.exists(position_dir):
        print(f"Directory {position_dir} does not exist. Skipping.")
        return

    for player_file in os.listdir(position_dir):
        player_path = os.path.join(position_dir, player_file)
        process_goalkeeper_player_file(player_path, season, last_season_dir)
# Function to process a single goalkeeper's file
def process_goalkeeper_player_file(player_path, season, last_season_dir):
    try:
        df = pd.read_csv(player_path)
    except pd.errors.EmptyDataError:
        print(f"File {player_path} is empty or corrupted. Skipping.")
        return
    
    required_columns = ['total_points', 'minutes', 'clean_sheets', 'saves']
    if not all(col in df.columns for col in required_columns):
        print(f"File {player_path} is missing required columns. Skipping.")
        return
    
    # 1. Create 'form' column if 'total_points' is available
    if 'total_points' in df.columns:
        df['form'] = calculate_goalkeeper_form(df['total_points'])
    
    # 2. Calculate clean sheet probability if 'minutes' and 'clean_sheets' are available
    if 'minutes' in df.columns and 'clean_sheets' in df.columns:
        df['clean_sheet_probability'] = df['clean_sheets'].expanding().apply(lambda x: calculate_clean_sheet_probability(df[:x.index[-1]+1]), raw=False)
    
    # 3. Calculate saves per game if 'saves' is available
    if 'saves' in df.columns:
        df['saves_per_game'] = calculate_saves_per_game(df)
    
    # 4. Get last season stats (only for 2022-23 and 2023-24 seasons)
    if season in ['2022-23', '2023-24', '2024-25']:
        player_file = os.path.basename(player_path)
        last_season_penalties_saved, last_season_expected_goals_conceded, last_season_clean_sheet_prob, last_season_total_saves = get_last_season_stats_goalkeeper(player_file, last_season_dir)
        df['last_season_penalties_saved'] = last_season_penalties_saved
        df['last_season_expected_goals_conceded'] = last_season_expected_goals_conceded
        df['last_season_clean_sheet_probability'] = last_season_clean_sheet_prob
        df['last_season_total_saves'] = last_season_total_saves
    
    # Round all numerical columns to two decimal places
    df = df.round(2)
    
    # Save the updated dataframe back to the CSV file
    df.to_csv(player_path, index=False)
    print(f"Processed and saved {player_path}")



#ADD THE NEXT WEEK GAMEWEEK POINTS TO EACH ROW
def add_next_gw_pts(base_dir='player_data', seasons=['2022-23', '2023-24', '2024-25']):
    
    positions=['FWD', 'MID', 'GK', 'DEF']; 
    """
    Processes player CSV files in the given directory structure by adding a 'next_week_points' column.
    
    Args:
    base_dir (str): The base directory containing the player data.
    seasons (list of str): List of season directories to process.
    positions (list of str): List of position directories to process.
    
    """
    # Loop through each season
    for season_index, season in enumerate(seasons):
        for position in positions:
            # Get the path to the position folder in the current season
            position_dir = os.path.join(base_dir, season, position)

            # Ensure the position directory exists
            if not os.path.exists(position_dir):
                print(f"Position directory {position_dir} does not exist. Skipping.")
                continue

            # Loop through each player CSV file in the position directory
            for player_file in os.listdir(position_dir):
                # Load the CSV file for the player
                player_file_path = os.path.join(position_dir, player_file)
                try:
                    df = pd.read_csv(player_file_path)
                except pd.errors.EmptyDataError:
                    print(f"File {player_file_path} is empty or corrupted. Skipping.")
                    continue

                # Check if the file has a 'total_points' column
                if 'total_points' not in df.columns:
                    print(f"File {player_file_path} does not have 'total_points'. Skipping.")
                    continue

                # Create a new column for next week's points
                df['next_week_points'] = None

                # Iterate through each row
                for i in range(len(df) - 1):
                    df.at[i, 'next_week_points'] = df.at[i + 1, 'total_points']

                # Handle the case for the last row, only if it's not the last season
                if season_index < len(seasons) - 1 and len(df) > 0:
                    # Look for the same player file in the next season, same position
                    next_season = seasons[season_index + 1]
                    next_season_dir = os.path.join(base_dir, next_season, position)
                    next_season_player_file = os.path.join(next_season_dir, player_file)

                    # Ensure the next season file exists and can be read
                    if os.path.exists(next_season_player_file):
                        try:
                            next_season_df = pd.read_csv(next_season_player_file)
                            if len(next_season_df) > 0:
                                # Get the first row's total_points from the next season
                                df.at[len(df) - 1, 'next_week_points'] = next_season_df.at[0, 'total_points']
                        except pd.errors.EmptyDataError:
                            print(f"File {next_season_player_file} is empty or corrupted. Skipping last row addition.")

                # Save the updated CSV file back to the same location
                df.to_csv(player_file_path, index=False)

    print("Processing completed.")

  

###ADDING FORM 
def map_team_name(team_name, team_name_mappings):
    """Map team names to their canonical versions."""
    return team_name_mappings.get(team_name, team_name)

def get_closest_match(team_name, options, threshold=75):
    """Get the closest match for a team name using fuzzywuzzy."""
    closest_match, score = process.extractOne(team_name, options)
    if score >= threshold:
        return closest_match
    return None

def get_difficulty_subdirectory(season, row_index):
    """Determine the appropriate difficulty directory based on the row index."""
    season_start_year = int(season.split('-')[0])
    if row_index <= 19:  # First half of the season
        return f'difficulty_{season_start_year}'
    else:  # Second half of the season
        return f'difficulty_half_{season_start_year+1}'

def get_next_fixture_difficulty(player_team, was_home, opponent_team, season, row_index, fixture_difficulties_dir, team_name_mappings):
    """Calculate next week's specific fixture difficulty."""
    mapped_team = map_team_name(player_team, team_name_mappings)
    difficulty_subdir = get_difficulty_subdirectory(season, row_index)
    difficulty_dir_path = os.path.join(fixture_difficulties_dir, difficulty_subdir)

    if os.path.exists(difficulty_dir_path):
        team_files = os.listdir(difficulty_dir_path)
        closest_team_file = get_closest_match(mapped_team, team_files)

        if closest_team_file:
            team_difficulty_file_path = os.path.join(difficulty_dir_path, closest_team_file)
            team_df = pd.read_csv(team_difficulty_file_path)
            opponent = get_closest_match(opponent_team, team_df['Opponent'].tolist())

            if opponent is not None:
                if was_home:
                    difficulty = team_df.loc[team_df['Opponent'] == opponent, 'Home Difficulty'].values
                else:
                    difficulty = team_df.loc[team_df['Opponent'] == opponent, 'Away Difficulty'].values
                if len(difficulty) > 0:
                    return round(difficulty[0], 2)
    return None

def get_next_holistic_fixture_difficulty(opponent_team, was_home, season, row_index, holistic_difficulties_dir, team_name_mappings):
    """Calculate holistic fixture difficulty for the next week."""
    mapped_team = map_team_name(opponent_team, team_name_mappings)
    difficulty_subdir = get_difficulty_subdirectory(season, row_index)
    difficulty_dir_path = os.path.join(holistic_difficulties_dir, difficulty_subdir)

    if os.path.exists(difficulty_dir_path):
        team_files = os.listdir(difficulty_dir_path)
        closest_team_file = get_closest_match(mapped_team, team_files)

        if closest_team_file:
            team_difficulty_file_path = os.path.join(difficulty_dir_path, closest_team_file)
            team_df = pd.read_csv(team_difficulty_file_path)

            if was_home:
                difficulty = team_df['Away Difficulty'].values
            else:
                difficulty = team_df['Home Difficulty'].values
            if len(difficulty) > 0:
                return round(difficulty[0], 2)
    return None

def get_next_season_fixture(player_file, current_season_index, position, seasons, merged_data_dir):
    """Handle moving to the next season for the last row."""
    if current_season_index < len(seasons) - 1:
        next_season = seasons[current_season_index + 1]
        next_season_dir = os.path.join(merged_data_dir, next_season, position)
        if os.path.exists(next_season_dir):
            next_season_files = os.listdir(next_season_dir)
            closest_player_file = get_closest_match(player_file, next_season_files, threshold=75)
            if closest_player_file:
                next_player_path = os.path.join(next_season_dir, closest_player_file)
                next_df = pd.read_csv(next_player_path)
                if len(next_df) > 0:
                    return next_df.iloc[0]['team'], next_df.iloc[0]['was_home'], next_df.iloc[0]['opponent_team']
    return None, None, None

def process_form_data(merged_data_dir, fixture_difficulties_dir, holistic_difficulties_dir, seasons, positions, team_name_mappings):
    """Main function to traverse the data and calculate fixture difficulties."""
    for season_index, season in enumerate(seasons):
        for position in positions:
            position_dir = os.path.join(merged_data_dir, season, position)
            
            if os.path.exists(position_dir):
                for player_file in os.listdir(position_dir):
                    player_file_path = os.path.join(position_dir, player_file)
                    
                    try:
                        df = pd.read_csv(player_file_path)
                    except pd.errors.EmptyDataError:
                        print(f"File {player_file_path} is empty or corrupted. Skipping.")
                        continue
                    
                    if df.empty or not {'team', 'was_home', 'opponent_team'}.issubset(df.columns):
                        continue

                    df['next_team'] = df['team'].shift(-1)
                    df['next_was_home'] = df['was_home'].shift(-1)
                    df['next_opponent_team'] = df['opponent_team'].shift(-1)

                    last_row_index = df.index[-1]
                    if season_index < len(seasons) - 1:
                        next_season_team, next_season_was_home, next_season_opponent_team = get_next_season_fixture(
                            player_file, season_index, position, seasons, merged_data_dir)
                        
                        if next_season_team:
                            df.loc[last_row_index, 'next_team'] = next_season_team
                            df.loc[last_row_index, 'next_was_home'] = next_season_was_home
                            df.loc[last_row_index, 'next_opponent_team'] = next_season_opponent_team
                    
                    df['next_week_specific_fixture_difficulty'] = df.apply(
                        lambda row: get_next_fixture_difficulty(
                            row['next_team'], row['next_was_home'], row['next_opponent_team'], season, row.name,
                            fixture_difficulties_dir, team_name_mappings
                        ) if pd.notnull(row['next_team']) and pd.notnull(row['next_opponent_team']) else None, axis=1
                    )

                    df['next_week_holistic_fixture_difficulty'] = df.apply(
                        lambda row: get_next_holistic_fixture_difficulty(
                            row['next_opponent_team'], row['next_was_home'], season, row.name,
                            holistic_difficulties_dir, team_name_mappings
                        ) if pd.notnull(row['next_opponent_team']) else None, axis=1
                    )

                    df.to_csv(player_file_path, index=False)
                    print(f"Updated {player_file_path} with next_week_specific_fixture_difficulty and next_week_holistic_fixture_difficulty.")
            else:
                print(f"Position directory {position_dir} does not exist for season {season}.")

def process_forms_default():
    """Helper function to call process_player_data with default directories and mappings."""
    merged_data_dir = 'player_data'
    fixture_difficulties_dir = 'fixture_difficulties'
    holistic_difficulties_dir = 'holistic_difficulties'

    seasons = ['2022-23', '2023-24', '2024-25']
    positions = ['GK', 'DEF', 'MID', 'FWD']

    team_name_mappings = {
        "Spurs": "Tottenham",
        "Man City": "Manchester City",
        "Man Utd": "Manchester United",
        "Nott'm Forest": "Nottingham Forest"
    }

    process_form_data(merged_data_dir, fixture_difficulties_dir, holistic_difficulties_dir, seasons, positions, team_name_mappings)



###CREATE TRAIN DATA 
def create_train_data(base_dir, train_data_dir, seasons):
    # Create the train_data directory if it doesn't exist
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)

    # Define the positions and corresponding output files
    positions_mapping = {
        'FWD': 'forward.csv',
        'MID': 'midfielder.csv',
        'GK': 'goalkeeper.csv',
        'DEF': 'defender.csv'
    }

    # Initialize empty dataframes for each position
    position_dfs = {
        'FWD': pd.DataFrame(),
        'MID': pd.DataFrame(),
        'GK': pd.DataFrame(),
        'DEF': pd.DataFrame()
    }

    # Loop through the seasons
    for season in seasons:
        # Loop through each position (FWD, MID, GK, DEF)
        for position, output_file in positions_mapping.items():
            position_dir = os.path.join(base_dir, season, position)

            # Ensure the position directory exists
            if not os.path.exists(position_dir):
                print(f"Position directory {position_dir} does not exist. Skipping.")
                continue

            # Loop through each player CSV file in the position directory
            for player_file in os.listdir(position_dir):
                player_file_path = os.path.join(position_dir, player_file)

                try:
                    df = pd.read_csv(player_file_path)
                except pd.errors.EmptyDataError:
                    print(f"File {player_file_path} is empty or corrupted. Skipping.")
                    continue

                # Filter rows where 'next_week_points' is defined (non-null or non-NaN)
                df_filtered = df[df['next_week_points'].notnull()]

                # Append filtered data to the corresponding dataframe for this position
                position_dfs[position] = pd.concat([position_dfs[position], df_filtered])

    # Save each position's dataframe into the corresponding CSV in train_data
    for position, output_file in positions_mapping.items():
        output_path = os.path.join(train_data_dir, output_file)
        position_dfs[position].to_csv(output_path, index=False)
        print(f"{output_file} saved with {len(position_dfs[position])} rows.")

    print("Processing completed.")

def clean_footprint(base_dirs=['fpl_gw_data', 'understat_data']):
    """
    Deletes all files and directories within the provided base directories.
    
    Parameters:
    - base_dirs: A list of base directories to clean (e.g., ['fpl_gw_data', 'understat_data']).
    """
    for base_dir in base_dirs:
        # Check if the directory exists
        if os.path.exists(base_dir):
            # Loop through all subdirectories and files in the base directory
            for root, dirs, files in os.walk(base_dir, topdown=False):
                # Remove files
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
                    except OSError as e:
                        print(f"Error deleting file {file_path}: {e}")

                # Remove directories
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Deleted directory: {dir_path}")
                    except OSError as e:
                        print(f"Error deleting directory {dir_path}: {e}")

            # Remove the base directory itself
            try:
                shutil.rmtree(base_dir)
                print(f"Deleted base directory: {base_dir}")
            except OSError as e:
                print(f"Error deleting base directory {base_dir}: {e}")
        else:
            print(f"Directory {base_dir} does not exist. Skipping.")






def merge_all(): 
  #merge_fpl_and_understat_data(); 
  #process_fwd_data(); 
  #process_mid_data(); 
  #process_def_data(); 
  #process_gk_data();  
  ###GET THE FORMS 
  #form_fwd_mid_data();
  #form_defender_data();
  #form_goalkeeper_data(); 
  ###ADD THE NEXT GW PTS 
  #add_next_gw_pts(); 
  ###ADD FORMS 
  #process_forms_default(); 
  create_train_data('player_data', 'train_data', ['2022-23', '2023-24']); 
  clean_footprint();



if __name__ == "__main__":
    merge_all()

