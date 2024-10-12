import pandas as pd
import requests
from io import StringIO

class PremierLeagueTeam:
    def __init__(self, team_name):
        self.team_name = team_name
        self.expected_goals = 0
        self.actual_goals = 0
        self.expected_goals_against = 0
        self.actual_goals_against = 0

    def __repr__(self):
        return (f"Team: {self.team_name}\n"
                f"Expected Goals (xG): {self.expected_goals}\n"
                f"Actual Goals: {self.actual_goals}\n"
                f"Expected Goals Against (xGA): {self.expected_goals_against}\n"
                f"Actual Goals Against: {self.actual_goals_against}\n")

    def update_stats(self, xg, goals, xga, goals_against):
        self.expected_goals += xg
        self.actual_goals += goals
        self.expected_goals_against += xga
        self.actual_goals_against += goals_against

def fetch_csv_data_from_github(url):
    """Fetch CSV data from a GitHub URL and return it as a pandas DataFrame."""
    response = requests.get(url)
    if response.status_code == 200:
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    else:
        raise Exception(f"Failed to fetch data from {url}")

def get_team_stats_for_season(season, teams):
    """Scrape the expected goals (xG), actual goals, expected goals against (xGA), and actual goals against for each team in a given season."""
    #Change the link 
    base_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/understat/merged_gw.csv"
    df = fetch_csv_data_from_github(base_url)
    
    # Filter relevant columns: team name, expected goals (xG), goals, expected goals against (xGA), and goals against
    relevant_columns = ['team', 'xG', 'goals_scored', 'xGA', 'goals_conceded']
    df_filtered = df[relevant_columns]
    
    for team in teams:
        team_data = df_filtered[df_filtered['team'] == team.team_name]
        team_xg = team_data['xG'].sum()
        team_goals = team_data['goals_scored'].sum()
        team_xga = team_data['xGA'].sum()
        team_goals_against = team_data['goals_conceded'].sum()

        team.update_stats(team_xg, team_goals, team_xga, team_goals_against)

def main():
    # List of teams
    team_names = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Liverpool', 'Luton', 'Man City', 'Man Utd', 'Newcastle', 'Nott\'m Forest', 'Sheffield Utd', 'Spurs', 'West Ham', 'Wolves']  # Add the team names you want
    teams = [PremierLeagueTeam(name) for name in team_names]
    
    # Get data for the last two seasons
    seasons = ['2022-23', '2023-24']
    
    for season in seasons:
        get_team_stats_for_season(season, teams)
    
    # Display team stats
    for team in teams:
        print(team)

if __name__ == "__main__":
    main()

