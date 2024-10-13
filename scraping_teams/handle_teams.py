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
        
        try:
            # Fetch the CSV file for the team
            df = fetch_csv_data_from_github(team_url)
            
            # Save the CSV data to a local file in the output folder
            local_file_path = os.path.join(output_folder, f"{team_filename}_{season}.csv")
            df.to_csv(local_file_path, index=False)
            
            print(f"Data for {team_name} saved successfully to {local_file_path}")
        
        except Exception as e:
            print(f"Error fetching data for {team_name}: {e}")

def main():
    # List of team names
    team_names = [
        'Arsenal', 'Aston_Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley',
        'Chelsea', 'Crystal_Palace', 'Everton', 'Fulham', 'Liverpool', 'Luton',
        'Manchester_City', 'Manchester_United', 'Newcastle_United', 'Nottingham_Forest', 'Sheffield_United', 'Southampton', 
        'Tottenham', 'West_Ham', 'Wolverhampton_Wanderers'
    ]
    
    # Specify seasons and the output folder for the CSV files
    seasons = ['2021-22', '2022-23', '2023-24']
    output_folder = 'fpl_team_data'  # Directory to save the CSV files
    
    for season in seasons:
        save_team_data_to_csv(season, team_names, output_folder)

if __name__ == "__main__":
    main()
