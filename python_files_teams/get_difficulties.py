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

def calculate_difficulty(team, opponent, location, team_fixtures, team_fpl_data):
    # Filter fixtures to find matches against the opponent
    opponent_matches = team_fixtures[(team_fixtures['Opponent'] == opponent) & (team_fixtures['Location'] == location)]

    if opponent_matches.empty:
        return None  # If there are no matches against the opponent for that location, return None
    
    # Gather difficulties based on normalized stats from FPL data
    difficulties = []

    for idx, match in opponent_matches.iterrows():
        # Find the corresponding row in the FPL data using the index
        try:
            fpl_row = team_fpl_data.iloc[idx]

            # Calculate weighted differences using the normalized data
            # (missed - scored) with weight 5
            missed_scored_diff = (fpl_row['missed_normalized'] - fpl_row['scored_normalized']) * 5

            # (xGA - xG) with weight 3
            xga_xg_diff = (fpl_row['xGA_normalized'] - fpl_row['xG_normalized']) * 3

            # (ppda_allowed - ppda) with weight 2
            ppda_diff = (fpl_row['ppda_allowed_normalized'] - fpl_row['ppda_normalized']) * 2

            # (deep - deep_allowed) with weight 2
            deep_diff = (fpl_row['deep_normalized'] - fpl_row['deep_allowed_normalized']) * 2

            # Sum up the difficulty based on the weighted differences
            total_difficulty = missed_scored_diff + xga_xg_diff + ppda_diff + deep_diff

            difficulties.append(total_difficulty)
        except IndexError:
            # If there’s no matching row in the FPL data, continue to the next match
            continue

    # If there were no valid matches, return None
    if not difficulties:
        return None

    # Return the average difficulty for the opponent
    return sum(difficulties) / len(difficulties)


def normalize_team_name(filename):
    # Split the filename into parts by underscore
    parts = filename.split('_')
    
    # Join all parts except the last one (which is the season)
    team_name = '_'.join(parts[:-1])
    
    # Replace underscores with spaces to get the proper team name
    return team_name.replace('_', ' ')

# Main function to calculate difficulties
def calculate_difficulties_for_all_teams(fpl_team_data_dir, fixtures_dir, output_dir, seasons):
    teams = set(normalize_team_name(filename) for filename in os.listdir(fixtures_dir) if filename.endswith('.csv'))

    for team in teams:
        print(f"Processing {team}...")

        difficulties = []  # To store home and away difficulties for all seasons
        for season in seasons:
            team_fixture_file = f"{team.replace(' ', '_')}_{season}.csv"
            fpl_team_data_file = f"{team.replace(' ', '_')}_{season}.csv"

            # Try loading fixture and FPL data for the season
            try:
                # Load the fixtures for the team
                team_fixtures = pd.read_csv(os.path.join(fixtures_dir, team_fixture_file))

                # Load the FPL data for the team
                team_fpl_data = pd.read_csv(os.path.join(fpl_team_data_dir, fpl_team_data_file))

                # Calculate difficulties for all opponents
                for opponent in teams:
                    if team != opponent:  # Don't calculate difficulty for itself
                        home_difficulty = calculate_difficulty(team, opponent, 'Home', team_fixtures, team_fpl_data)
                        away_difficulty = calculate_difficulty(team, opponent, 'Away', team_fixtures, team_fpl_data)

                        difficulties.append({
                            'Opponent': opponent,
                            'Home Difficulty': home_difficulty,
                            'Away Difficulty': away_difficulty
                        })
            except FileNotFoundError:
                print(f"File not found for {team} in {season}. Skipping this season.")
                continue  # Move to the next season

        # Calculate average difficulty across available seasons
        if difficulties:
            difficulties_df = pd.DataFrame(difficulties)
            # Group by opponent and average across seasons
            average_difficulties = difficulties_df.groupby('Opponent').mean().reset_index()

            # Save difficulty ratings for the team
            output_file = os.path.join(output_dir, f"{team}_difficulty_ratings.csv")
            average_difficulties.to_csv(output_file, index=False)
            print(f"Saved difficulty ratings for {team} to {output_file}")


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

# Set weights for different factors
weight_dict = {
    'scored_missed': 5,
    'xG_xGA': 3,
    'ppda_diff': 2,
    'deep_diff': 2
}

# Example usage
fpl_team_data_dir = './fpl_team_data'
fixtures_dir = './fixturesForEachTeam'
output_dir = './output'
seasons = ['2021-22', '2022-23', '2023-24']

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

calculate_difficulties_for_all_teams(fpl_team_data_dir, fixtures_dir, output_dir, seasons)
