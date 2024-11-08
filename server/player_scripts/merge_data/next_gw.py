import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def add_next_gw_points(base_dir='player_data', seasons=['2022-23', '2023-24', '2024-25'], positions=['FWD', 'MID', 'GK', 'DEF']):
    """
    Processes player CSV files in the given directory structure by adding a 'next_week_points' column.

    Args:
    base_dir (str): The base directory containing the player data.
    seasons (list of str): List of season directories to process.
    positions (list of str): List of position directories to process.
    """
    def process_file_for_next_gw(player_file_path, next_season_file_path=None):
        try:
            df = pd.read_csv(player_file_path)
        except pd.errors.EmptyDataError:
            print(f"File {player_file_path} is empty or corrupted. Skipping.")
            return

        if 'total_points' not in df.columns or 'date' not in df.columns:
            print(f"File {player_file_path} does not have required columns. Skipping.")
            return

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['next_week_points'] = df['total_points'].shift(-1)

        # Handle the last row if there is a next season file and the date is in May
        if next_season_file_path and not df.empty and pd.notna(df.at[len(df) - 1, 'date']):
            last_row_date = df.at[len(df) - 1, 'date']
            if last_row_date.month == 5:  # Check if the last row's date is in May
                try:
                    next_season_df = pd.read_csv(next_season_file_path)
                    if 'total_points' in next_season_df.columns and not next_season_df.empty:
                        df.at[len(df) - 1, 'next_week_points'] = next_season_df.at[0, 'total_points']
                except pd.errors.EmptyDataError:
                    return
                    #print(f"File {next_season_file_path} is empty or corrupted. Skipping last row addition.")

        df.to_csv(player_file_path, index=False)
        print(f"Processed and saved {player_file_path}")

    with ThreadPoolExecutor() as executor:
        for season_index, season in enumerate(seasons):
            for position in positions:
                position_dir = os.path.join(base_dir, season, position)

                if not os.path.exists(position_dir):
                    print(f"Position directory {position_dir} does not exist. Skipping.")
                    continue

                next_season_dir = None
                if season_index < len(seasons) - 1:
                    next_season_dir = os.path.join(base_dir, seasons[season_index + 1], position)

                player_files = [os.path.join(position_dir, player_file) for player_file in os.listdir(position_dir) if player_file.endswith('.csv')]

                for player_file_path in player_files:
                    next_season_file_path = None
                    if next_season_dir:
                        next_season_file_path = os.path.join(next_season_dir, os.path.basename(player_file_path))
                        if not os.path.exists(next_season_file_path):
                            next_season_file_path = None

                    executor.submit(process_file_for_next_gw, player_file_path, next_season_file_path)

    print("Processing completed.")

# Example usage
if __name__ == "__main__":
    add_next_gw_points()