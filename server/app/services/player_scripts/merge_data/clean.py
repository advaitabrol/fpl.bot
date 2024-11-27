import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def process_and_sort_data(merged_data_dir='player_data', seasons=['2021-22', '2022-23', '2023-24', '2024-25']):
    positions = ['FWD', 'MID', 'DEF', 'GK']

    # Define columns to drop and the desired order for each position
    column_config = {
        'FWD': {
            'columns_to_drop': ['position_x', 'position_y', 'time', 'kickoff_time'],
            'desired_column_order': [
                'name', 'team', 'opponent_team', 'date', 'was_home', 'minutes', 'goals', 'xG', 'assists', 'xA',
                'total_points', 'shots', 'key_passes', 'ict_index', 'bonus', 'starts', 'selected'
            ]
        },
        'MID': {
            'columns_to_drop': ['position_x', 'position_y', 'own_goals', 'time', 'kickoff_time'],
            'desired_column_order': [
                'name', 'team', 'opponent_team', 'date', 'was_home', 'minutes', 'goals', 'xG', 'assists', 'xA',
                'total_points', 'shots', 'key_passes', 'ict_index', 'bonus', 'starts', 'selected'
            ]
        },
        'DEF': {
            'columns_to_drop': ['position_x', 'time', 'kickoff_time', 'own_goals'],
            'desired_column_order': [
                'name', 'team', 'opponent_team', 'date', 'was_home', 'position', 'minutes', 'goals', 'xG', 'assists', 'xA',
                'total_points', 'shots', 'key_passes', 'ict_index', 'bonus', 'clean_sheets', 'goals_conceded', 'expected_goals_conceded', 'starts', 'selected'
            ]
        },
        'GK': {
            'columns_to_drop': ['position_x', 'time', 'kickoff_time', 'own_goals', 'goals', 'shots', 'xG', 'assists', 'key_passes', 'position_y'],
            'desired_column_order': [
                'name', 'team', 'opponent_team', 'date', 'was_home', 'minutes', 'goals_conceded', 'expected_goals_conceded',
                'saves', 'penalties_saved', 'total_points', 'bonus', 'clean_sheets', 'xA', 'starts', 'selected'
            ]
        }
    }

    def process_csv_file(season, position, csv_file):
        file_path = os.path.join(merged_data_dir, season, position, csv_file)
        df = pd.read_csv(file_path)

        # Step 1: Sort by 'kickoff_time' if it exists
        if 'kickoff_time' in df.columns:
            df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')
            df = df.sort_values(by='kickoff_time').dropna(subset=['kickoff_time'])

        # Step 2: Drop unnecessary columns
        columns_to_drop = column_config[position]['columns_to_drop']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Step 3: Reorder columns
        desired_column_order = column_config[position]['desired_column_order']
        existing_columns = [col for col in desired_column_order if col in df.columns]
        df = df[existing_columns]

        # Save the processed CSV file
        df.to_csv(file_path, index=False)
        print(f"Processed and saved {csv_file} in {os.path.join(merged_data_dir, season, position)}")

    # Process CSV files for each season and position concurrently
    for season in seasons:
        for position in positions:
            position_dir = os.path.join(merged_data_dir, season, position)
            if os.path.isdir(position_dir):
                csv_files = [f for f in os.listdir(position_dir) if f.endswith('.csv')]
                with ThreadPoolExecutor() as executor:
                    executor.map(lambda csv_file: process_csv_file(season, position, csv_file), csv_files)
            else:
                print(f"{position} directory for season {season} does not exist.")

# Example usage
if __name__ == "__main__":
    process_and_sort_data()