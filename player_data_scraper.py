import os
import requests
import pandas as pd
import shutil


#SCRAPING FPL_GW DATA --> IN GW FORMAT 
def download_fpl_gw_csv_files(base_url='https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/', seasons=['2021-22', '2022-23', '2023-24', '2024-25'], gw_range=range(1, 39)):
    """
    Downloads gameweek CSV files from the raw GitHub URL for specified seasons and gameweeks,
    and saves them in a directory in the current working directory.

    Parameters:
    base_url (str): The base URL for the raw GitHub content (e.g., 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/').
    seasons (list): A list of seasons (e.g., ['2021-22', '2022-23', '2023-24']).
    gw_range (range): The range of gameweeks to download (e.g., range(1, 39)).
    """
    # Get current working directory (CWD)
    cwd = os.getcwd()
    # Create a master folder named gw_data inside the CWD
    master_folder = os.path.join(cwd, 'gw_data')

    # Create master folder if it doesn't exist
    if not os.path.exists(master_folder):
        os.makedirs(master_folder)

    for season in seasons:
        # Create season folder inside master folder
        season_folder = os.path.join(master_folder, season)
        if not os.path.exists(season_folder):
            os.makedirs(season_folder)

        for gw in gw_range:
            gw_file = f'gw{gw}.csv'
            gw_url = f'{base_url}{season}/gws/{gw_file}'

            # Download the CSV file from the raw GitHub URL
            response = requests.get(gw_url)

            # Check if the request was successful
            if response.status_code == 200:
                # Save the file to the appropriate folder
                with open(os.path.join(season_folder, gw_file), 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {gw_file} for {season}")
            else:
                print(f"Failed to download {gw_file} for {season}: {response.status_code} (URL: {gw_url})")



#ADJUSTING FPL GW DATA --> PLAYER FORMAT
def fpl_gw_to_player():
  # Define the paths for the original and new directories
  base_gw_directory = os.path.join(os.getcwd(), 'gw_data')
  new_directory = os.path.join(os.getcwd(), 'fpl_gw_data')
  # Create the new directory if it doesn't exist
  if not os.path.exists(new_directory):
    os.makedirs(new_directory)
  # Traverse all season folders in fpl_gw_data
  for season in sorted(os.listdir(base_gw_directory)):
    season_path = os.path.join(base_gw_directory, season)
    if os.path.isdir(season_path):
      print(f"Processing season: {season}")
      process_season_fpl_gw(season, base_gw_directory, new_directory)
 # Delete the fpl_raw_data directory after processing
  if os.path.exists(base_gw_directory):
      shutil.rmtree(base_gw_directory)
      print(f"Deleted directory: {base_gw_directory}")
  

def process_season_fpl_gw(season, basedir, newdir):
    season_folder = os.path.join(basedir, season)
    new_season_folder = os.path.join(newdir, season)
    
    # Create season subdirectory in the new directory if it doesn't exist
    if not os.path.exists(new_season_folder):
        os.makedirs(new_season_folder)
    
    # Dictionary to hold player data
    player_data = {}

    # Iterate over all gw{x}.csv files (gameweek files)
    for gw_file in sorted(os.listdir(season_folder)):
        if gw_file.endswith('.csv'):
            gw_path = os.path.join(season_folder, gw_file)
            df = pd.read_csv(gw_path)

            # Iterate through each row (player entry) in the gameweek file
            for index, row in df.iterrows():
                player_name = row['name']
                
                # If player_name is not in the dictionary, initialize an empty list for the player
                if player_name not in player_data:
                    player_data[player_name] = []

                # Append the row data (converted to a dictionary) to the player's list
                player_data[player_name].append(row)

    # Now save each player's data into their respective CSV files
    for player_name, rows in player_data.items():
        player_df = pd.DataFrame(rows)  # Convert the list of rows to a DataFrame
        player_csv_filename = f"{player_name.replace(' ', '_')}.csv"
        player_csv_path = os.path.join(new_season_folder, player_csv_filename)

        # Save the DataFrame to the player's CSV file
        player_df.to_csv(player_csv_path, index=False)



#CLEAN THE UNECESSARY COLUMNS IN THE FPL GW DATA
def clean_fpl_gw_data(): 
  # Define the path to the fpl_gw_data directory
  fpl_gw_data_dir = 'fpl_gw_data'
  # Define the columns to drop based on position
  columns_to_drop_by_position = {
      'FWD': ['clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved', 'saves', 'expected_goals_conceded'],
      'MID': ['own_goals', 'penalties_saved', 'saves', 'expected_goals_conceded'],
      'DEF': ['saves', 'penalties_saved'],
      'GK': ['expected_goals', 'goals_scored', 'ict_index', 'penalties_missed']
  }
  # Define the columns to always drop
  columns_to_always_drop = [
      'bps', 'creativity', 'element', 'expected_goal_involvements', 'influence', 
      'round', 'selected', 'team_a_score', 'team_h_score', 
      'threat', 'transfers_balance', 'transfers_in', 'transfers_out', 'value', 'expected_goals', 'goals_scored', 'assists', 'expected_assists', 
      'penalties_missed', 'red_cards', 'yellow_cards', 'fixture', 'xP'
  ]
  # Define team names and IDs for each season based on alphabetical order
  TEAM_IDS = {
      '2021-22': {
          1: 'Arsenal', 2: 'Aston Villa', 3: 'Brentford', 4: 'Brighton', 5: 'Burnley',
          6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton', 9: 'Leeds United', 10: 'Leicester City',
          11: 'Liverpool', 12: 'Manchester City', 13: 'Manchester United', 14: 'Newcastle United', 
          15: 'Norwich', 16: 'Southampton', 17: 'Tottenham', 18: 'Watford', 19: 'West Ham', 20: 'Wolverhampton'
      },
      '2022-23': {
          1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford', 5: 'Brighton', 
          6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton', 9: 'Fulham', 10: 'Leeds United', 
          11: 'Leicester City', 12: 'Liverpool', 13: 'Manchester City', 14: 'Manchester United', 
          15: 'Newcastle United', 16: 'Nottingham Forest', 17: 'Southampton', 18: 'Tottenham', 
          19: 'West Ham', 20: 'Wolverhampton'
      },
      '2023-24': {
          1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford', 5: 'Brighton', 6: 'Burnley',
          7: 'Chelsea', 8: 'Crystal Palace', 9: 'Everton', 10: 'Fulham', 11: 'Liverpool', 
          12: 'Luton', 13: 'Manchester City', 14: 'Manchester United', 15: 'Newcastle United', 
          16: 'Nottingham Forest', 17: 'Sheffield', 18: 'Tottenham', 19: 'West Ham', 20: 'Wolverhampton'
      },
      '2024-25': {
        1: "Arsenal",
        2: "Aston Villa",
        3: "Bournemouth",
        4: "Brentford",
        5: "Brighton",
        6: "Chelsea",
        7: "Crystal Palace",
        8: "Everton",
        9: "Fulham",
        10: "Ipswich Town",
        11: "Leicester City",
        12: "Liverpool",
        13: "Manchester City",
        14: "Manchester United",
        15: "Newcastle United",
        16: "Nottingham Forest",
        17: "Southampton",
        18: "Tottenham",
        19: "West Ham",
        20: "Wolverhampton"
      }
  }

  # Traverse through each season and each player CSV file
  for season in os.listdir(fpl_gw_data_dir):
      season_dir = os.path.join(fpl_gw_data_dir, season)
  
      if os.path.isdir(season_dir):
          for gw_file in os.listdir(season_dir):
              if gw_file.endswith('.csv'):
                  file_path = os.path.join(season_dir, gw_file)
                  
                  # Read the CSV file
                  df = pd.read_csv(file_path)
                  
                  if 'position' in df.columns:
                      # Get the position from the first non-header row
                      first_position = df['position'].iloc[0]  # Assuming the first row has valid data
                      
                      # Drop columns based on the position
                      columns_to_drop = columns_to_drop_by_position.get(first_position, [])
                      df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
                      
                      # Drop the columns that should always be removed
                      df_cleaned = df_cleaned.drop(columns=[col for col in columns_to_always_drop if col in df_cleaned.columns], errors='ignore')
                      
                      # Map opponent_team ID to team name
                      if 'opponent_team' in df_cleaned.columns:
                          df_cleaned['opponent_team'] = df_cleaned['opponent_team'].map(TEAM_IDS[season])
                      
                      # Modify the kickoff_time column to only show the date (YYYY-MM-DD)
                      if 'kickoff_time' in df_cleaned.columns:
                          df_cleaned['kickoff_time'] = pd.to_datetime(df_cleaned['kickoff_time']).dt.date
                      
                      # Save the modified dataframe back to the CSV file
                      df_cleaned.to_csv(file_path, index=False)



###SCRAPING DATA FROM UNDERSTAT
def download_understat_csv_files(base_url='https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/season/understat/', seasons=['2021-22', '2022-23', '2023-24', '2024-25']):
    """
    Downloads all CSV files from the understat GitHub directory for the specified seasons.
    Saves them in a directory in the current working directory.

    Parameters:
    base_url (str): The base URL for the raw GitHub content (e.g., 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/season/understat/').
    seasons (list): A list of seasons (e.g., ['2021-22', '2022-23', '2023-24']).
    """
    # Get current working directory (CWD)
    cwd = os.getcwd()
    # Create a master folder named understat_data inside the CWD
    master_folder = os.path.join(cwd, 'understat_data')

    # Create master folder if it doesn't exist
    if not os.path.exists(master_folder):
        os.makedirs(master_folder)

    for season in seasons:
        # Create season folder inside master folder
        season_folder = os.path.join(master_folder, season)
        if not os.path.exists(season_folder):
            os.makedirs(season_folder)

        # Corrected GitHub API URL to list all files in the season's understat directory
        api_url = f'https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data/{season}/understat/'

        # Get list of files in the directory
        response = requests.get(api_url)
        if response.status_code == 200:
            files = response.json()  # Parse the JSON response
            for file in files:
                if file['name'].endswith('.csv'):
                    file_url = file['download_url']
                    file_name = file['name']

                    # Download the CSV file from the raw GitHub URL
                    file_response = requests.get(file_url)
                    if file_response.status_code == 200:
                        # Save the file to the appropriate folder
                        with open(os.path.join(season_folder, file_name), 'wb') as f:
                            f.write(file_response.content)
                        print(f"Downloaded {file_name} for {season}")
                    else:
                        print(f"Failed to download {file_name} for {season}: {file_response.status_code}")
        else:
            print(f"Failed to retrieve file list for {season}: {response.status_code}")



###CLEANING THE UNDERSTAT DATA
def clean_understat():
  understat_dir = 'understat_data'

  # Define the columns to drop after filtering rows by date
  columns_to_drop = ['h_team', 'a_team', 'h_goals', 'a_goals', 'id', 'season', 'roster_id', 'npg', 'npxG', 'xGChain', 'xGBuildup']

  # Define the columns to aggregate
  #columns_to_aggregate = ['goals', 'shots', 'xG', 'time', 'xA', 'assists', 'key_passes']

  # Define the valid date ranges for each season
  season_date_ranges = {
      '2024-25': ('2024-08-01', '2025-05-31'),
      '2023-24': ('2023-08-01', '2024-05-31'),
      '2022-23': ('2022-08-01', '2023-05-31'),
      '2021-22': ('2021-08-01', '2022-05-31') 

  }

  # Traverse through each season and each player CSV file
  for season in os.listdir(understat_dir):
      season_dir = os.path.join(understat_dir, season)
      
      if os.path.isdir(season_dir):
          # Get the date range for the current season
          start_date, end_date = season_date_ranges.get(season, (None, None))
          
          if start_date and end_date:
              for player_file in os.listdir(season_dir):
                  if player_file.endswith('.csv'):
                      file_path = os.path.join(season_dir, player_file)
                      
                      # Read the player CSV file
                      df = pd.read_csv(file_path)
                      
                      # Check if the 'date' column exists before attempting to filter by date
                      if 'date' in df.columns:
                          # Filter rows by valid date range for the current season
                          df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime format
                          df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                      else:
                          # If 'date' column doesn't exist, skip filtering
                          df_filtered = df
                      
                      # Now drop the unwanted columns, only if they exist in the DataFrame
                      df_cleaned = df_filtered.drop(columns=[col for col in columns_to_drop if col in df_filtered.columns], errors='ignore')
                      
                      # Filter only the columns that exist in the DataFrame
                      '''
                      existing_columns_to_aggregate = [col for col in columns_to_aggregate if col in df_cleaned.columns]
                      
                      if existing_columns_to_aggregate:  # Ensure there are columns to aggregate
                          # Calculate the aggregate sums for the existing columns
                          aggregate_row = df_cleaned[existing_columns_to_aggregate].sum()

                          # Convert the aggregate row to a DataFrame and append it
                          aggregate_row_df = pd.DataFrame([aggregate_row])
                          df_cleaned = pd.concat([df_cleaned, aggregate_row_df], ignore_index=True)
                          
                          # Calculate the stats per 90 minutes
                          if 'time' in aggregate_row and aggregate_row['time'] > 0:
                              total_minutes = aggregate_row['time']
                              per_90_row = aggregate_row / (total_minutes / 90)
                          else:
                              per_90_row = aggregate_row
                              
                          # Ensure total_minutes remains in the row
                          per_90_row['time'] = total_minutes if 'time' in aggregate_row else 0

                          # Convert the per-90-minutes row to a DataFrame and append it
                          per_90_row_df = pd.DataFrame([per_90_row])
                          df_cleaned = pd.concat([df_cleaned, per_90_row_df], ignore_index=True)
                      '''
                      # Save the modified dataframe back to the CSV file
                      df_cleaned.to_csv(file_path, index=False)


def scrape_all(): 
  ###get the FPL GW data
  download_fpl_gw_csv_files(); #DOWNLOADS WEEK TO WEEK FPL DATA
  fpl_gw_to_player(); #CHANGES FPL DATA TO PLAYER FORMAT
  clean_fpl_gw_data(); 
  ###get the understat data
  download_understat_csv_files(); 
  clean_understat(); 
    
if __name__ == "__main__":
    scrape_all()