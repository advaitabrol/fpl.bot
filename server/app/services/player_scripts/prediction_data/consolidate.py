import os
import pandas as pd

def combine_csv_files(gw_dirname: str):
    # Define paths
    base_dir = os.getcwd()
    season_folder = os.path.join(base_dir, "prediction_data", "2024-25", gw_dirname)
    
    # Define file paths for each position
    positions = {
        "GK": "gk.csv",
        "DEF": "def.csv",
        "MID": "mid.csv",
        "FWD": "fwd.csv"
    }

    # Initialize a list to store dataframes
    dataframes = []

    # Loop through each position file
    for position, filename in positions.items():
        file_path = os.path.join(season_folder, filename)
        
        # Read the CSV file if it exists
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['position'] = position  # Add the position column
            dataframes.append(df)  # Append the dataframe to the list
        else:
            print(f"Warning: {filename} does not exist in {season_folder}")

    # Concatenate all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Write the combined dataframe to all.csv
        output_dir = '../client/public'
        output_path = os.path.join(output_dir, "all.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Combined CSV file created at {output_path}")
    else:
        print("No CSV files were found to combine.")

if __name__ == "__main__": 
    combine_csv_files("Gw_10")