import os
import pandas as pd
import ast


"""
This goes through the files with each teams match specific data in each season and normalizes 
all of it. Important to do before calculating difficulties. 
"""

# Function to parse 'ppda' and 'ppda_allowed' fields
def parse_ppda(ppda_str):
    try:
        ppda_dict = ast.literal_eval(ppda_str)  # Safely parse the string to a dictionary
        return ppda_dict['att'] / ppda_dict['def']  # Ratio of 'att' and 'def'
    except (ValueError, SyntaxError, KeyError):
        return None  # Return None if the data is invalid

# Min-Max normalization function
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Directory for team data
fpl_team_data_dir = './fpl_team_data/'

# Seasons to normalize: 2019-20 and 2020-21
seasons_to_normalize = ['2019-20', '2020-21']

# Process each file in the directory
for filename in os.listdir(fpl_team_data_dir):
    if filename.endswith('.csv'):
        # Extract the season from the filename (format: teamname_season.csv)
        season = filename.split('_')[-1].split('.')[0]
        
        # Only process files for 2019-20 and 2020-21 seasons
        if season in seasons_to_normalize:
            file_path = os.path.join(fpl_team_data_dir, filename)
            
            # Load team data
            df = pd.read_csv(file_path)

            # Ensure the columns exist before normalizing
            if {'scored', 'missed', 'xG', 'xGA', 'ppda', 'ppda_allowed', 'deep', 'deep_allowed'}.issubset(df.columns):
                
                # Parse 'ppda' and 'ppda_allowed' as ratios of att and def
                df['ppda'] = df['ppda'].apply(parse_ppda)
                df['ppda_allowed'] = df['ppda_allowed'].apply(parse_ppda)

                # Perform normalization for relevant columns
                df['scored_normalized'] = min_max_normalize(df['scored'])
                df['missed_normalized'] = min_max_normalize(df['missed'])
                df['xG_normalized'] = min_max_normalize(df['xG'])
                df['xGA_normalized'] = min_max_normalize(df['xGA'])
                df['ppda_normalized'] = min_max_normalize(df['ppda'])
                df['ppda_allowed_normalized'] = min_max_normalize(df['ppda_allowed'])
                df['deep_normalized'] = min_max_normalize(df['deep'])
                df['deep_allowed_normalized'] = min_max_normalize(df['deep_allowed'])

                # Save the modified DataFrame back to CSV with the normalized values
                df.to_csv(file_path, index=False)
                print(f"Normalized data and saved for: {filename}")
            else:
                print(f"File {filename} does not have the required columns.")
        else:
            print(f"Skipping normalization for {filename} as it belongs to {season}.")
