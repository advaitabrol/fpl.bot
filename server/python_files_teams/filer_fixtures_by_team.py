import pandas as pd
import os

"""
This file iterates through the files created by fetch_fixture_list.py and separates the 
fixtures by team. The output is a folder which holds files indicating team and season. 
Each file has the list of all fixutres in order for the indicated team and season, holding
home team, away team, home team score, away team score, and goal difference. 
"""

# List of seasons to iterate through
seasons = ['2024-25']

# Directory where your fixture files are stored
fixtures_dir = './fpl_season_fixtures_info/'

# Directory to save the team-specific fixtures
output_dir = './team_fixtures/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through each season
for season in seasons:
    # Load the fixtures data for the current season
    fixtures_file = os.path.join(fixtures_dir, f'fixtures_{season}.csv')
    fixtures = pd.read_csv(fixtures_file)

    # Create a set of all unique teams from the fixtures
    all_teams = set(fixtures['team_a']).union(set(fixtures['team_h']))

    # Iterate through each team to extract their fixtures
    for team in all_teams:
        # Filter the fixtures for the current team
        team_fixtures = fixtures[(fixtures['team_a'] == team) | (fixtures['team_h'] == team)]

        # Create an empty list to hold the fixture information for the team
        team_fixture_data = []

        # Iterate through the filtered fixtures and format the data for output
        for index, row in team_fixtures.iterrows():
            if row['team_a'] == team:
                # Team is away
                opponent = row['team_h']
                location = 'Away'
                team_score = row['team_a_score']
                opponent_score = row['team_h_score']
            else:
                # Team is home
                opponent = row['team_a']
                location = 'Home'
                team_score = row['team_h_score']
                opponent_score = row['team_a_score']

            # Calculate goal differential
            goal_diff = team_score - opponent_score

            # Append the fixture information
            team_fixture_data.append({
                'Opponent': opponent,
                'Location': location,
                'Team Score': team_score,
                'Opponent Score': opponent_score,
                'Goal Differential': goal_diff
            })

        # Convert the list of fixtures to a DataFrame
        team_df = pd.DataFrame(team_fixture_data)

        # Save the DataFrame to a CSV file
        team_df.to_csv(os.path.join(output_dir, f'{team}_{season}.csv'), index=False)

print("CSV files created for each team and season.")

