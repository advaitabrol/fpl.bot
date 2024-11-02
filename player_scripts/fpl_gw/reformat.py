import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def fpl_gw_to_player(base_gw_directory=None, new_directory=None):
    """
    Transforms gameweek CSV data into player-specific CSV files for each season.

    Parameters:
    base_gw_directory (str): The path to the base gameweek directory. Defaults to an environment variable or 'gw_data'.
    new_directory (str): The path to the new directory for player CSVs. Defaults to an environment variable or 'fpl_gw_data'.
    """
    # Set directories to environment variables if provided, or use default paths
    if base_gw_directory is None:
        base_gw_directory = os.getenv('FPL_GW_BASE_DIR', 'gw_data')
    if new_directory is None:
        new_directory = os.getenv('FPL_PLAYER_DATA_DIR', 'fpl_gw_data')

    # Create the new directory if it doesn't exist
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # Traverse all season folders in the base directory
    for season in sorted(os.listdir(base_gw_directory)):
        season_path = os.path.join(base_gw_directory, season)
        if os.path.isdir(season_path):
            print(f"Processing season: {season}")
            process_season_fpl_gw(season, base_gw_directory, new_directory)

def process_season_fpl_gw(season, basedir, newdir):
    season_folder = os.path.join(basedir, season)
    new_season_folder = os.path.join(newdir, season)

    if not os.path.exists(new_season_folder):
        os.makedirs(new_season_folder)

    def process_gameweek_file(gw_file):
        try:
            gw_path = os.path.join(season_folder, gw_file)
            df = pd.read_csv(gw_path)

            for _, row in df.iterrows():
                player_name = row['name']
                player_csv_filename = f"{player_name.replace(' ', '_')}.csv"
                player_csv_path = os.path.join(new_season_folder, player_csv_filename)

                pd.DataFrame([row]).to_csv(player_csv_path, mode='a', header=not os.path.exists(player_csv_path), index=False)
                print(f"Written data for {player_name} to {player_csv_path}")
        except Exception as e:
            print(f"Error processing {gw_file}: {e}")

    # Switch to ThreadPoolExecutor for I/O-bound tasks
    with ThreadPoolExecutor() as executor:
        gw_files = [gw_file for gw_file in sorted(os.listdir(season_folder)) if gw_file.endswith('.csv')]
        executor.map(process_gameweek_file, gw_files)

    print(f"Completed processing for season: {season}")

# Example usage of running the function (for server usage)
'''
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    fpl_gw_to_player()
'''