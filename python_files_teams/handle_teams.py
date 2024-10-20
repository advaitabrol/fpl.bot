import os
import pandas as pd
import requests
from io import StringIO

def fetch_csv_data_from_github(url):
    """Fetch CSV data from a GitHub URL and return it as a pandas DataFrame."""
    response = requests.get(url)
    if response.status_code == 200:
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    else:
        raise Exception(f"Failed to fetch data from {url}")

def save_team_data_to_csv(season, team_names, output_folder):
    """Fetch each team's data from the GitHub repository and save it as a CSV file locally."""
    base_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/understat/"
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for team_name in team_names:
        # Convert team names to the file format by replacing spaces with underscores
        team_filename = team_name.replace(' ', '_')
        team_url = base_url + f"understat_{team_filename}.csv"
        
        # Define the local file path
        local_file_path = os.path.join(output_folder, f"{team_filename}_{season}.csv")
        
        # Check if the file already exists to avoid rescraping
        if os.path.exists(local_file_path):
            print(f"Data for {team_name} in {season} already exists. Skipping...")
            continue
        
        try:
            # Fetch the CSV file for the team
            df = fetch_csv_data_from_github(team_url)
            
            # Save the CSV data to a local file in the output folder
            df.to_csv(local_file_path, index=False)
            
            print(f"Data for {team_name} saved successfully to {local_file_path}")
        
        except Exception as e:
            print(f"Error fetching data for {team_name} in {season}: {e}")

def main():
    # Teams for each season
    team_names_per_season = {
        '2017-18': [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Burnley', 'Chelsea', 'Crystal Palace',
            'Everton', 'Huddersfield', 'Leicester', 'Liverpool', 'Manchester City', 
            'Manchester United', 'Newcastle United', 'Southampton', 'Stoke', 'Swansea', 
            'Tottenham', 'Watford', 'West Brom', 'West Ham'
        ],
        '2018-19': [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Burnley', 'Chelsea', 'Crystal Palace', 
            'Everton', 'Fulham', 'Huddersfield', 'Leicester', 'Liverpool', 'Manchester City', 
            'Manchester United', 'Newcastle United', 'Southampton', 'Tottenham', 'Watford', 
            'Wolverhampton Wanderers', 'West Ham', 'Cardiff'
        ],
        '2019-20': [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Burnley', 'Chelsea', 'Crystal Palace', 
            'Everton', 'Leicester', 'Liverpool', 'Manchester City', 'Manchester United', 
            'Newcastle United', 'Norwich', 'Sheffield United', 'Southampton', 'Tottenham', 
            'Watford', 'West Ham', 'Wolverhampton Wanderers', 'Brighton'
        ],
        '2020-21': [
            'Arsenal', 'Aston Villa', 'Brighton', 'Burnley', 'Chelsea', 'Crystal Palace', 
            'Everton', 'Fulham', 'Leeds', 'Leicester', 'Liverpool', 'Manchester City', 
            'Manchester United', 'Newcastle United', 'Sheffield United', 'Southampton', 
            'Tottenham', 'West Brom', 'West Ham', 'Wolverhampton Wanderers'
        ],
        '2021-22': [
            'Arsenal', 'Aston Villa', 'Brentford', 'Brighton', 'Burnley', 'Chelsea', 
            'Crystal Palace', 'Everton', 'Leeds', 'Leicester', 'Liverpool', 
            'Manchester City', 'Manchester United', 'Newcastle United', 'Norwich', 
            'Southampton', 'Tottenham', 'Watford', 'West Ham', 'Wolverhampton Wanderers'
        ],
        '2022-23': [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea', 
            'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Leicester', 
            'Liverpool', 'Manchester City', 'Manchester United', 'Newcastle United', 
            'Nottingham Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolverhampton Wanderers'
        ],
        '2023-24': [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley', 'Chelsea', 
            'Crystal Palace', 'Everton', 'Fulham', 'Liverpool', 'Luton', 'Manchester City', 
            'Manchester United', 'Newcastle United', 'Nottingham Forest', 'Sheffield United', 
            'Tottenham', 'West Ham', 'Wolverhampton Wanderers'
        ]
    }
    
    # Specify the output folder for the CSV files
    output_folder = 'fpl_team_data'  # Directory to save the CSV files
    
    # Loop over each season and fetch the team data
    for season, team_names in team_names_per_season.items():
        save_team_data_to_csv(season, team_names, output_folder)

if __name__ == "__main__":
    main()
