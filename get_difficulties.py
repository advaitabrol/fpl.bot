import os
import pandas as pd
import ast

# Define weights for each season
season_weights = {'2023-24': 3, '2022-23': 2, '2021-22': 1}

# Directories for fixtures and team data
fixtures_dir = './fixturesForEachTeam/'
fpl_team_data_dir = './fpl_team_data/'
output_dir = './team_difficulty_ratings/'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to parse the ppda and ppda_allowed fields (which are saved as dictionary-like strings)
def parse_ppda(ppda_str):
    try:
        ppda_dict = ast.literal_eval(ppda_str)
        return ppda_dict['att'] + ppda_dict['def']  # Return the sum of att and def values
    except (ValueError, SyntaxError, KeyError):
        return None  # If parsing fails, return None

# Normalize team names (handle underscores and capitalization)
def normalize_team_name(name):
    return name.replace('_', ' ').strip().lower()

# Normalize the location data to match 'Home' and 'Away'
def normalize_location(location):
    return location.strip().capitalize()

# Function to extract the correct team name from the filename
def extract_team_name(filename):
    # Split the filename by underscores to get parts
    parts = filename.split('_')
    
    # If there are 2 underscores, the team name consists of two words (e.g., "West_Ham")
    if len(parts) == 3:  # Example: "West_Ham_2023-24.csv"
        team_name = ' '.join(parts[:2])  # Join the first two parts as the team name
    else:
        team_name = parts[0]  # Use the first part for single-word team names
    
    # Special case for Wolverhampton Wanderers (we only want 'Wolverhampton')
    if team_name == "Wolverhampton Wanderers":
        team_name = "Wolverhampton"
    
    return team_name.strip().lower()  # Normalize and return the name

# Get list of teams based on the file names in the fixtures folder
teams = list(set(extract_team_name(filename) for filename in os.listdir(fixtures_dir) if filename.endswith('.csv')))

# Function to calculate difficulty based on actual outcomes and stats
def calculate_difficulty(team, opponent, home_away, fixture_data, team_stats, weight):
    difficulties = []
    
    # Ensure both files contain the same number of games (rows should correspond)
    if len(fixture_data) == len(team_stats):  
        for idx, match in fixture_data.iterrows():
            stat = team_stats.iloc[idx]  # Align stats row with fixture row
            location = normalize_location(match['Location'])
            if location == home_away:
                # Extract relevant stats
                scored = stat['scored']
                missed = stat['missed']
                ppda = parse_ppda(stat['ppda'])  # Parse ppda
                ppda_allowed = parse_ppda(stat['ppda_allowed'])  # Parse ppda_allowed
                deep = stat['deep']
                deep_allowed = stat['deep_allowed']

                # Handle None values in ppda, ppda_allowed
                if ppda is None or ppda_allowed is None:
                    print(f"Warning: Missing PPDA data for {team} vs {opponent}, skipping this match.")
                    continue  # Skip if PPDA data is invalid

                # Combine stats for difficulty
                difficulty = ((scored - missed) + (ppda - ppda_allowed) + (deep - deep_allowed)) * weight
                difficulties.append(difficulty)
                print(f"Calculated difficulty for {team} vs {opponent} ({home_away}): {difficulty}")
    
    # Calculate the average difficulty (weighing recent seasons more)
    return sum(difficulties) / len(difficulties) if difficulties else None

# Iterate over each team to calculate home and away difficulties
for team in teams:
    for season, weight in season_weights.items():
        # Normalize the team name for comparison and file name construction
        team_file_name = team.replace(' ', '_')
        
        # Construct file paths for fixture and fpl team data
        team_fixtures_file = os.path.join(fixtures_dir, f'{team_file_name}_{season}.csv')
        team_fpl_data_file = os.path.join(fpl_team_data_dir, f'{team_file_name}_{season}.csv')

        print(f"First few rows of fixture data for {team} in season {season}:")
        print(team_fixtures.head())  # Print first few rows to inspect for errors

        print(f"First few rows of FPL data for {team} in season {season}:")
        print(team_fpl_data.head())  # Print first few rows to inspect for errors
        
        # Check if the files exist
        if not os.path.exists(team_fixtures_file):
            print(f"Fixture file missing for {team} in season {season}")
        if not os.path.exists(team_fpl_data_file):
            print(f"FPL data file missing for {team} in season {season}")
        
        if os.path.exists(team_fixtures_file) and os.path.exists(team_fpl_data_file):
            # Load the fixture and fpl team data, skipping the first row (header)
            team_fixtures = pd.read_csv(team_fixtures_file, skiprows=1)  # Skip header row
            team_fpl_data = pd.read_csv(team_fpl_data_file, skiprows=1)  # Skip header row
            
            # Debugging: Print number of games and check if the lengths match
            print(f"Processing {team} for season {season}")
            print(f"Fixture rows: {len(team_fixtures)}, FPL data rows: {len(team_fpl_data)}")
            
            # Create a DataFrame to store difficulty ratings
            difficulties = []
            
            for opponent in teams:
                print(f"Processing match: Team = {team}, Opponent = {opponent}")  # Debugging: Check if team and opponent are matching correctly
                if team != opponent:  # Skip if the team is the same
                    # Calculate home difficulty
                    home_difficulty = calculate_difficulty(team, opponent, 'Home', team_fixtures, team_fpl_data, weight)
                    # Calculate away difficulty
                    away_difficulty = calculate_difficulty(team, opponent, 'Away', team_fixtures, team_fpl_data, weight)

                    # Store the result
                    difficulties.append({
                        'Team': team,
                        'Opponent': opponent,
                        'Home Difficulty': home_difficulty,
                        'Away Difficulty': away_difficulty
                    })
                    # Debugging: Check if the difficulty is being computed correctly
                    print(f"{team} vs {opponent}: Home: {home_difficulty}, Away: {away_difficulty}")

            # Save the difficulty ratings to a CSV file
            difficulty_df = pd.DataFrame(difficulties)
            difficulty_df.to_csv(os.path.join(output_dir, f'{team_file_name}_difficulty_ratings.csv'), index=False)

print("Difficulty ratings generated and saved.")
