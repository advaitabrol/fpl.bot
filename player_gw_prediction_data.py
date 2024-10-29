import os
import pandas as pd


def create_prediction_data(base_dir, prediction_dir):
    """
    This method goes into the player_data/2024-25 season, navigates each subdirectory (GK, DEF, MID, FWD),
    copies the second last row from each player.csv file, and stores the data into gk.csv, def.csv, mid.csv, fwd.csv
    in the prediction_data/2024-25/GW #x directory, where x is the index of the second last row.
    
    Parameters:
    - base_dir: The base directory where the player data is stored (e.g., player_data).
    - prediction_dir: The directory where the prediction data will be saved (e.g., prediction_data).
    """

    # Define the positions and corresponding output files
    positions_mapping = {
        'GK': 'gk.csv',
        'DEF': 'def.csv',
        'MID': 'mid.csv',
        'FWD': 'fwd.csv'
    }

    season_dir = os.path.join(base_dir, '2024-25')
    
    # Ensure the season directory exists
    if not os.path.exists(season_dir):
        print(f"Season directory {season_dir} does not exist.")
        return
    
    # Initialize empty dataframes for each position
    position_dfs = {
        'GK': pd.DataFrame(),
        'DEF': pd.DataFrame(),
        'MID': pd.DataFrame(),
        'FWD': pd.DataFrame()
    }
    
    second_last_row_index = None
    
    # Loop through each position directory (GK, DEF, MID, FWD)
    for position, output_file in positions_mapping.items():
        position_dir = os.path.join(season_dir, position)

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

            # Get the second last row of the dataframe (assuming at least two rows exist)
            if len(df) >= 2:
                second_last_row = df.iloc[[-2]]  # Get the second last row
                position_dfs[position] = pd.concat([position_dfs[position], second_last_row])

                # Set the index for creating the GW folder if not already set
                if second_last_row_index is None:
                    second_last_row_index = df.index[-2]  # Get the index of the second last row
        
    if second_last_row_index is None:
        print("No valid data found in player CSV files.")
        return
    
    # Create the prediction_data directory and season subdirectory if they don't exist
    prediction_season_dir = os.path.join(prediction_dir, '2024-25', f'GW #{second_last_row_index}')
    os.makedirs(prediction_season_dir, exist_ok=True)

    # Save each position's dataframe into the corresponding CSV in the prediction_data directory
    for position, output_file in positions_mapping.items():
        output_path = os.path.join(prediction_season_dir, output_file)
        position_dfs[position].to_csv(output_path, index=False)
        print(f"{output_file} saved in {prediction_season_dir} with {len(position_dfs[position])} rows.")

    print("Prediction data processing completed.")

# Example usage
create_prediction_data('player_data', 'prediction_data')
