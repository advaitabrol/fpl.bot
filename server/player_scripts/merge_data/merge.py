import os
import pandas as pd
from fuzzywuzzy import process, fuzz
from concurrent.futures import ThreadPoolExecutor

def merge_fpl_and_understat_data(seasons=['2021-22', '2022-23', '2023-24', '2024-25'], merged_data_dir='player_data', fpl_gw_data_dir='fpl_gw_data', understat_data_dir='understat_data'):
    positions = ['FWD', 'MID', 'DEF', 'GK']
    
    # Create the merged_data directory with subdirectories for each season and position
    for season in seasons:
        season_dir = os.path.join(merged_data_dir, season)
        os.makedirs(season_dir, exist_ok=True)
        
        for position in positions:
            position_dir = os.path.join(season_dir, position)
            os.makedirs(position_dir, exist_ok=True)

    # Function to find the best match using fuzzy matching
    def find_best_match(player_name, understat_files):
        best_match = process.extractOne(player_name, understat_files, scorer=fuzz.partial_ratio)
        if best_match and best_match[1] >= 25:  # Adjust the threshold as needed
            return best_match[0]
        return None

    # Function to process each FPL CSV file
    def process_fpl_file(fpl_csv, season, fpl_season_dir, understat_season_dir):
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

    # Use ThreadPoolExecutor for concurrent processing of CSV files
    for season in seasons:
        fpl_season_dir = os.path.join(fpl_gw_data_dir, season)
        understat_season_dir = os.path.join(understat_data_dir, season)
        
        if os.path.isdir(fpl_season_dir) and os.path.isdir(understat_season_dir):
            fpl_csv_files = [f for f in os.listdir(fpl_season_dir) if f.endswith('.csv')]
            with ThreadPoolExecutor() as executor:
                executor.map(lambda fpl_csv: process_fpl_file(fpl_csv, season, fpl_season_dir, understat_season_dir), fpl_csv_files)
        else:
            print(f"Season folder {season} does not exist in both {fpl_gw_data_dir} and {understat_data_dir}.")

# Example usage
if __name__ == "__main__":
    merge_fpl_and_understat_data()