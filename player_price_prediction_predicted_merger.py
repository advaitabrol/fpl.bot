import os
import pandas as pd
from fuzzywuzzy import process

def add_price_to_prediction_data(prediction_dir, prices_file, threshold=60):
    """
    Goes through each CSV file in the prediction_data/2024-25 folder, and for each row, it adds a 'price' column
    by matching the 'name' to the closest name in the current_prices.csv using fuzzy matching.

    Parameters:
    - prediction_dir: The directory containing the CSV files (e.g., 'prediction_data/2024-25').
    - prices_file: The path to the current_prices.csv file.
    - threshold: The minimum fuzzy match score to consider a name match.
    """

    # Load the current_prices.csv file
    prices_df = pd.read_csv(prices_file)
    
    # Ensure that the necessary columns are present in the current_prices.csv
    if 'name' not in prices_df.columns or 'now_cost' not in prices_df.columns:
        print(f"Required columns ('name', 'now_cost') not found in {prices_file}")
        return

    # Convert the prices_df into a dictionary for fast lookup by name
    prices_dict = dict(zip(prices_df['name'], prices_df['now_cost']))

    # Go into the prediction_data/2024-25 directory
    for root, dirs, files in os.walk(prediction_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)

                # Read each CSV file
                df = pd.read_csv(file_path)
                
                if 'name' not in df.columns:
                    print(f"'name' column not found in {file_path}. Skipping file.")
                    continue

                # Initialize a 'price' column
                df['price'] = None

                # Iterate over each row in the CSV and find the closest name match using fuzzy matching
                for index, row in df.iterrows():
                    player_name = row['name']
                    
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

def add_status_data(): 
    return; 
# Example usage
prediction_dir = 'prediction_data/2024-25'
prices_file = 'current_prices.csv'
add_price_to_prediction_data(prediction_dir, prices_file)