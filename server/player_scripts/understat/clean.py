import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def clean_understat_data(understat_dir='understat_data'):
    """
    Cleans and filters Understat data CSV files for each season by removing unnecessary columns,
    filtering by date range, and ensuring data quality.

    Parameters:
    understat_dir (str): The directory containing Understat data. Defaults to 'understat_data'.
    """
    # Columns to drop
    columns_to_drop = ['h_team', 'a_team', 'h_goals', 'a_goals', 'id', 'season', 'roster_id', 'npg', 'npxG', 'xGChain', 'xGBuildup']

    # Valid date ranges for each season
    season_date_ranges = {
        '2024-25': ('2024-08-01', '2025-05-31'),
        '2023-24': ('2023-08-01', '2024-05-31'),
        '2022-23': ('2022-08-01', '2023-05-31'),
        '2021-22': ('2021-08-01', '2022-05-31')
    }

    def clean_player_file(season, player_file):
        try:
            file_path = os.path.join(understat_dir, season, player_file)
            df = pd.read_csv(file_path)

            # Ensure the 'date' column exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert 'date' to datetime, handling errors
                df = df.dropna(subset=['date'])  # Remove rows where 'date' conversion failed
                start_date, end_date = season_date_ranges.get(season, (None, None))
                
                if start_date and end_date:
                    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                else:
                    df_filtered = df
            else:
                df_filtered = df

            # Drop specified columns if they exist
            df_cleaned = df_filtered.drop(columns=[col for col in columns_to_drop if col in df_filtered.columns], errors='ignore')
            
            # Save the cleaned data back to the CSV
            df_cleaned.to_csv(file_path, index=False)
            print(f"Cleaned {player_file} for season {season}")
        except Exception as e:
            print(f"Error cleaning {player_file} in season {season}: {e}")

    # Iterate through each season and process player files concurrently
    for season in os.listdir(understat_dir):
        season_dir = os.path.join(understat_dir, season)
        if os.path.isdir(season_dir):
            print(f"Processing season: {season}")
            player_files = [f for f in os.listdir(season_dir) if f.endswith('.csv')]
            with ThreadPoolExecutor() as executor:
                executor.map(lambda f: clean_player_file(season, f), player_files)

    print("Finished cleaning all Understat data.")

# Example usage
'''
if __name__ == "__main__":
    clean_understat_data()
'''