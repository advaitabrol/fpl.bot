import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def create_train_data(base_dir='player_data', train_data_dir='train_data', seasons=['2022-23', '2023-24']):
    """
    Create training data by merging player data from specified seasons into positional CSVs.

    Args:
        base_dir (str): Base directory containing player data.
        train_data_dir (str): Directory where the output CSVs will be saved.
        seasons (list): List of season directories to process.
    """
    # Create the train_data directory if it doesn't exist
    os.makedirs(train_data_dir, exist_ok=True)

    # Define the positions and corresponding output files
    positions_mapping = {
        'FWD': 'forward.csv',
        'MID': 'midfielder.csv',
        'GK': 'goalkeeper.csv',
        'DEF': 'defender.csv'
    }

    # Initialize empty dataframes for each position
    position_dfs = {position: [] for position in positions_mapping}

    def process_position_file(player_file_path, position):
        """Read and filter a player CSV, appending non-null rows to position data."""
        try:
            df = pd.read_csv(player_file_path)
            # Filter rows where 'next_week_points' is defined
            df_filtered = df[df['next_week_points'].notnull()]
            if not df_filtered.empty:
                position_dfs[position].append(df_filtered)
        except pd.errors.EmptyDataError:
            print(f"File {player_file_path} is empty or corrupted. Skipping.")

    # Loop through the seasons
    with ThreadPoolExecutor() as executor:
        for season in seasons:
            for position, output_file in positions_mapping.items():
                position_dir = os.path.join(base_dir, season, position)

                # Ensure the position directory exists
                if not os.path.exists(position_dir):
                    print(f"Position directory {position_dir} does not exist. Skipping.")
                    continue

                # Process each player CSV file in the position directory concurrently
                player_files = [os.path.join(position_dir, f) for f in os.listdir(position_dir) if f.endswith('.csv')]
                for player_file_path in player_files:
                    executor.submit(process_position_file, player_file_path, position)

    # Concatenate and save each position's dataframe into the corresponding CSV in train_data
    for position, output_file in positions_mapping.items():
        output_path = os.path.join(train_data_dir, output_file)
        if position_dfs[position]:
            final_df = pd.concat(position_dfs[position], ignore_index=True)
            final_df.to_csv(output_path, index=False)
            print(f"{output_file} saved with {len(final_df)} rows.")
        else:
            print(f"No data for position {position}. {output_file} not created.")

    print("Processing completed.")

if __name__ == "__main__":
    create_train_data()