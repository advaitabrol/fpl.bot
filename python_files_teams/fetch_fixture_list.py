import os
import pandas as pd
import requests
from io import StringIO

# Define team names and IDs for the 2024-25 season
TEAM_IDS = {
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

def fetch_csv_data_from_github(url):
    """Fetch CSV data from a GitHub URL and return it as a pandas DataFrame."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch data from {url}: {e}")

def replace_team_ids_with_names(fixtures_df, season):
    """Replace the team_a and team_h IDs in the fixtures DataFrame with actual team names using predefined mapping."""
    team_mapping = TEAM_IDS[season]
    fixtures_df['team_a'] = fixtures_df['team_a'].map(team_mapping)
    fixtures_df['team_h'] = fixtures_df['team_h'].map(team_mapping)
    return fixtures_df

def process_fixtures_for_season(season, fixtures_csv_url, output_folder):
    """Process the fixtures for a given season, replacing team IDs with team names and saving filtered data."""
    try:
        # Fetch fixtures.csv for the season
        print(f"Fetching data for season {season} from {fixtures_csv_url}")
        fixtures_df = fetch_csv_data_from_github(fixtures_csv_url)
        
        # Replace team IDs with team names
        fixtures_df = replace_team_ids_with_names(fixtures_df, season)
        
        # Select relevant columns: team_a, team_a_score, team_h, team_h_score
        relevant_columns = ['team_a', 'team_a_score', 'team_h', 'team_h_score']
        fixtures_filtered_df = fixtures_df[relevant_columns]
        
        # Save the filtered data to a new CSV file
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        output_file = os.path.join(output_folder, f"fixtures_{season}.csv")
        fixtures_filtered_df.to_csv(output_file, index=False)
        
        print(f"Data for season {season} saved successfully to {output_file}")
    
    except Exception as e:
        print(f"Error processing data for season {season}: {e}")

def main():
    # Only the 2024-25 season to process
    season = '2024-25'
    
    # Base URL for fetching the file
    base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/"
    
    # URL for fixtures.csv for the 2024-25 season
    fixtures_csv_url = f"{base_url}{season}/fixtures.csv"
    
    # Specify the output folder where the processed CSV file will be saved
    output_folder = 'fpl_fixtures_data'
    
    # Process the fixtures for the 2024-25 season
    process_fixtures_for_season(season, fixtures_csv_url, output_folder)

if __name__ == "__main__":
    main()
