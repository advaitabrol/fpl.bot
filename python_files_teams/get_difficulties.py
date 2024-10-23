import os
import pandas as pd
import numpy as np

# Function to calculate average difficulty across seasons
def calculate_average_difficulty(team, opponent, difficulties):
    home_difficulties = [d['Home'] for d in difficulties if d['Home'] is not None]
    away_difficulties = [d['Away'] for d in difficulties if d['Away'] is not None]
    
    # Calculate averages, if there are values to average
    home_avg = np.mean(home_difficulties) if home_difficulties else None
    away_avg = np.mean(away_difficulties) if away_difficulties else None
    
    return home_avg, away_avg

# Function to calculate difficulty using normalized data with specific weights
def calculate_difficulty_from_normalized_data(fpl_row, weight_dict):
    # Extract normalized values
    scored_norm = fpl_row['scored_normalized']
    missed_norm = fpl_row['missed_normalized']
    xG_norm = fpl_row['xG_normalized']
    xGA_norm = fpl_row['xGA_normalized']
    ppda_norm = fpl_row['ppda_normalized']
    ppda_allowed_norm = fpl_row['ppda_allowed_normalized']
    deep_norm = fpl_row['deep_normalized']
    deep_allowed_norm = fpl_row['deep_allowed_normalized']
    
    # Apply weights
    difficulty = (
        weight_dict['scored_missed'] * (missed_norm - scored_norm) +
        weight_dict['xG_xGA'] * (xGA_norm - xG_norm) +
        weight_dict['ppda_diff'] * (ppda_allowed_norm - ppda_norm) +
        weight_dict['deep_diff'] * (deep_norm - deep_allowed_norm)
    )
    
    return difficulty

# Function to calculate difficulty for specific fixtures
def calculate_difficulty(team, opponent, location, team_fixtures, team_fpl_data, weight_dict):
    opponent_matches = team_fixtures[(team_fixtures['Opponent'] == opponent) & (team_fixtures['Location'] == location)]

    if opponent_matches.empty:
        return None  # If no matches against the opponent, return None
    
    difficulties = []
    for idx, match in opponent_matches.iterrows():
        try:
            fpl_row = team_fpl_data.iloc[idx]
            total_difficulty = calculate_difficulty_from_normalized_data(fpl_row, weight_dict)
            difficulties.append(total_difficulty)
        except IndexError:
            continue

    return sum(difficulties) / len(difficulties) if difficulties else None

# Helper function to normalize team name from file names
def normalize_team_name(filename):
    parts = filename.split('_')
    team_name = '_'.join(parts[:-1])
    return team_name.replace('_', ' ')

# Function to load team fixtures, handling partial season rows
def load_team_fixtures(fpl_team_data_dir, fixtures_dir, team, season, partial=False):
    team_fixture_file = f"{team.replace(' ', '_')}_{season}.csv"
    fpl_team_data_file = f"{team.replace(' ', '_')}_{season}.csv"

    # Load fixtures
    team_fixtures = pd.read_csv(os.path.join(fixtures_dir, team_fixture_file))
    
    # Load FPL data
    team_fpl_data = pd.read_csv(os.path.join(fpl_team_data_dir, fpl_team_data_file))
    
    # If partial season, limit to first 19 rows
    if partial:
        team_fixtures = team_fixtures.head(19)
        team_fpl_data = team_fpl_data.head(19)
    
    return team_fixtures, team_fpl_data

# Function to calculate difficulties for all teams and all seasons
def calculate_difficulties_for_all_teams(fpl_team_data_dir, fixtures_dir, output_dir, seasons, partial=False):
    # Define weight dictionary
    weight_dict = {
        'scored_missed': 5,
        'xG_xGA': 3,
        'ppda_diff': 2,
        'deep_diff': 2
    }

    teams = set(normalize_team_name(filename) for filename in os.listdir(fixtures_dir) if filename.endswith('.csv'))

    for team in teams:
        print(f"Processing {team}...")

        difficulties = []  # To store home and away difficulties across seasons
        for season in seasons:
            try:
                team_fixtures, team_fpl_data = load_team_fixtures(fpl_team_data_dir, fixtures_dir, team, season, partial)

                for opponent in teams:
                    if team != opponent:
                        home_difficulty = calculate_difficulty(team, opponent, 'Home', team_fixtures, team_fpl_data, weight_dict)
                        away_difficulty = calculate_difficulty(team, opponent, 'Away', team_fixtures, team_fpl_data, weight_dict)

                        difficulties.append({
                            'Opponent': opponent,
                            'Home Difficulty': home_difficulty,
                            'Away Difficulty': away_difficulty
                        })
            except FileNotFoundError:
                print(f"File not found for {team} in {season}. Skipping this season.")
                continue

        if difficulties:
            difficulties_df = pd.DataFrame(difficulties)
            average_difficulties = difficulties_df.groupby('Opponent').mean().reset_index()

            output_file = os.path.join(output_dir, f"{team}_difficulty_ratings.csv")
            average_difficulties.to_csv(output_file, index=False)
            print(f"Saved difficulty ratings for {team} to {output_file}")

# Main function to calculate difficulties for all requested scenarios
def main():
    # Base directories
    fpl_team_data_dir = './fpl_team_data'
    fixtures_dir = './fixturesForEachTeam'

    # Season sets for different calculations
    season_sets = {
        'difficulty_2019_2022': ['2019-20', '2020-21', '2021-22'],
        'difficulty_2019_2022_first_half_2022_23': ['2019-20', '2020-21', '2021-22', '2022-23'],
        'difficulty_2019_2023': ['2019-20', '2020-21', '2021-22', '2022-23'],
        'difficulty_2019_2023_first_half_2023_24': ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24'],
        'difficulty_2019_2024': ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
    }

    # Loop through each season set and calculate difficulties
    for folder_name, seasons in season_sets.items():
        output_dir = f"./output/{folder_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Check if partial season calculation is required
        partial = 'first_half' in folder_name
        calculate_difficulties_for_all_teams(fpl_team_data_dir, fixtures_dir, output_dir, seasons, partial=partial)

if __name__ == "__main__":
    main()
