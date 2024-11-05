import os
import pandas as pd
from fuzzywuzzy import process

def add_price_to_prediction_data(gw_folder, prices_file, threshold=60):
    """
    Adds a 'price' column to the def.csv, gk.csv, mid.csv, and fwd.csv files in the specified GW folder
    by matching the 'name' column to the closest name in current_prices.csv using fuzzy matching.

    Parameters:
    - gw_folder: The specific gameweek folder path (e.g., 'prediction_data/2024-25/GW10').
    - prices_file: The path to the current_prices.csv file.
    - threshold: The minimum fuzzy match score to consider a name match.
    """

    # Load the current_prices.csv file
    prices_df = pd.read_csv(prices_file)

    # Ensure that the necessary columns are present in the current_prices.csv
    if 'name' not in prices_df.columns or 'now_cost' not in prices_df.columns:
        print(f"Required columns ('name', 'now_cost') not found in {prices_file}")
        return

    # Ensure 'name' column in prices_df is of type str
    prices_df['name'] = prices_df['name'].astype(str)

    # Convert the prices_df into a dictionary for fast lookup by name
    prices_dict = dict(zip(prices_df['name'], prices_df['now_cost']))

    # Target files within the GW folder
    target_files = ['def.csv', 'gk.csv', 'mid.csv', 'fwd.csv']

    for file_name in target_files:
        file_path = os.path.join(gw_folder, file_name)

        # Check if the file exists before processing
        if not os.path.exists(file_path):
            print(f"{file_name} does not exist in {gw_folder}. Skipping.")
            continue

        # Read each CSV file
        df = pd.read_csv(file_path)

        if 'name' not in df.columns:
            print(f"'name' column not found in {file_path}. Skipping file.")
            continue

        # Ensure 'name' column in df is of type str and handle NaNs
        df['name'] = df['name'].fillna('').astype(str)

        # Initialize a 'price' column
        df['price'] = None

        # Iterate over each row in the CSV and find the closest name match using fuzzy matching
        for index, row in df.iterrows():
            player_name = row['name']

            if player_name:  # Check that player_name is not an empty string
                # Use fuzzywuzzy to find the closest match for the player name
                closest_match, score = process.extractOne(player_name, prices_dict.keys())

                # If the match score is above the threshold, assign the corresponding price
                if score >= threshold:
                    df.at[index, 'price'] = prices_dict[closest_match]
                else:
                    print(f"No suitable match found for '{player_name}' in {file_path} (closest match: '{closest_match}', score: {score})")

        # Save the updated CSV with the 'price' column added
        df.to_csv(file_path, index=False)
        print(f"Updated {file_path} with 'price' column.")

