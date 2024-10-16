import os
import pandas as pd

# Step 1: Define paths to the folders
fixtures_path = 'fixturesForEachTeam/'
fpl_data_path = 'fpl_team_data/'

# Step 2: Define a function to read CSV files
def read_csv_files(folder_path):
    csv_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            team_name = filename.split('_')[0]
            season = filename.split('_')[1].replace('.csv', '')
            df = pd.read_csv(os.path.join(folder_path, filename))
            csv_data[(team_name, season)] = df
    return csv_data

# Step 3: Load all the fixture and FPL team data
fixtures_data = read_csv_files(fixtures_path)
fpl_team_data = read_csv_files(fpl_data_path)

# Step 4: Define a function to extract useful statistics and calculate difficulty
def calculate_difficulty(fpl_team_df, fixtures_df):
    difficulties = {'home': {}, 'away': {}}

    # Iterate through all games in the season
    for index, fixture in fixtures_df.iterrows():
        # Find the corresponding game in the fpl_team_data based on chronological order
        game = fpl_team_df.iloc[index]

        opponent = fixture['Opponent']
        location = fixture['Location']
        team_score = fixture['Team Score']
        opponent_score = fixture['Opponent Score']
        goal_diff = fixture['Goal Differential']

        # Extract relevant stats from fpl_team_data
        xG = game['xG']
        xGA = game['xGA']
        scored = game['scored']
        missed = game['missed']
        result = game['result']  # 'w', 'l', 'd'
        points = game['pts']

        # Step 5: Calculate difficulty metric (you can customize this as per your needs)
        # Basic approach: high xG, goals scored, wins, and points = lower difficulty
        # More goals missed, high xGA, and losses = higher difficulty
        
        difficulty = (xGA + missed + (opponent_score - team_score) * 0.5) - (xG + scored + points * 0.5)

        # Store difficulty by home/away for each opponent
        if location == 'Home':
            if opponent not in difficulties['home']:
                difficulties['home'][opponent] = []
            difficulties['home'][opponent].append(difficulty)
        else:
            if opponent not in difficulties['away']:
                difficulties['away'][opponent] = []
            difficulties['away'][opponent].append(difficulty)

    return difficulties

# Step 6: Process data for each team and each season
team_difficulties = {}

for (team, season), fixtures_df in fixtures_data.items():
    if (team, season) in fpl_team_data:
        fpl_team_df = fpl_team_data[(team, season)]
        difficulties = calculate_difficulty(fpl_team_df, fixtures_df)

        # Store results
        if team not in team_difficulties:
            team_difficulties[team] = {'home': {}, 'away': {}}

        # Add home and away difficulty for the team
        for opponent in difficulties['home']:
            team_difficulties[team]['home'][opponent] = sum(difficulties['home'][opponent]) / len(difficulties['home'][opponent])

        for opponent in difficulties['away']:
            team_difficulties[team]['away'][opponent] = sum(difficulties['away'][opponent]) / len(difficulties['away'][opponent])

# Step 7: Output results for each team
for team, diff_data in team_difficulties.items():
    home_df = pd.DataFrame(list(diff_data['home'].items()), columns=['Opponent', 'Home Difficulty'])
    away_df = pd.DataFrame(list(diff_data['away'].items()), columns=['Opponent', 'Away Difficulty'])

    # Save each team's home and away difficulties to CSV
    home_df.to_csv(f'{team}_home_difficulty.csv', index=False)
    away_df.to_csv(f'{team}_away_difficulty.csv', index=False)

print("Difficulty ratings calculated and saved!")
import os
import pandas as pd

# Step 1: Define paths to the folders
fixtures_path = 'fixturesForEachTeam/'
fpl_data_path = 'fpl_team_data/'

# Step 2: Define a function to read CSV files
def read_csv_files(folder_path):
    csv_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            team_name = filename.split('_')[0]
            season = filename.split('_')[1].replace('.csv', '')
            df = pd.read_csv(os.path.join(folder_path, filename))
            csv_data[(team_name, season)] = df
    return csv_data

# Step 3: Load all the fixture and FPL team data
fixtures_data = read_csv_files(fixtures_path)
fpl_team_data = read_csv_files(fpl_data_path)

# Step 4: Define a function to extract useful statistics and calculate difficulty
def calculate_difficulty(fpl_team_df, fixtures_df):
    difficulties = {'home': {}, 'away': {}}

    # Iterate through all games in the season
    for index, fixture in fixtures_df.iterrows():
        # Find the corresponding game in the fpl_team_data based on chronological order
        game = fpl_team_df.iloc[index]

        opponent = fixture['Opponent']
        location = fixture['Location']
        team_score = fixture['Team Score']
        opponent_score = fixture['Opponent Score']
        goal_diff = fixture['Goal Differential']

        # Extract relevant stats from fpl_team_data
        xG = game['xG']
        xGA = game['xGA']
        scored = game['scored']
        missed = game['missed']
        result = game['result']  # 'w', 'l', 'd'
        points = game['pts']

        # Step 5: Calculate difficulty metric (you can customize this as per your needs)
        # Basic approach: high xG, goals scored, wins, and points = lower difficulty
        # More goals missed, high xGA, and losses = higher difficulty
        
        difficulty = (xGA + missed + (opponent_score - team_score) * 0.5) - (xG + scored + points * 0.5)

        # Store difficulty by home/away for each opponent
        if location == 'Home':
            if opponent not in difficulties['home']:
                difficulties['home'][opponent] = []
            difficulties['home'][opponent].append(difficulty)
        else:
            if opponent not in difficulties['away']:
                difficulties['away'][opponent] = []
            difficulties['away'][opponent].append(difficulty)

    return difficulties

# Step 6: Process data for each team and each season
team_difficulties = {}

for (team, season), fixtures_df in fixtures_data.items():
    if (team, season) in fpl_team_data:
        fpl_team_df = fpl_team_data[(team, season)]
        difficulties = calculate_difficulty(fpl_team_df, fixtures_df)

        # Store results
        if team not in team_difficulties:
            team_difficulties[team] = {'home': {}, 'away': {}}

        # Add home and away difficulty for the team
        for opponent in difficulties['home']:
            team_difficulties[team]['home'][opponent] = sum(difficulties['home'][opponent]) / len(difficulties['home'][opponent])

        for opponent in difficulties['away']:
            team_difficulties[team]['away'][opponent] = sum(difficulties['away'][opponent]) / len(difficulties['away'][opponent])

# Step 7: Output results for each team
for team, diff_data in team_difficulties.items():
    home_df = pd.DataFrame(list(diff_data['home'].items()), columns=['Opponent', 'Home Difficulty'])
    away_df = pd.DataFrame(list(diff_data['away'].items()), columns=['Opponent', 'Away Difficulty'])

    # Save each team's home and away difficulties to CSV
    home_df.to_csv(f'{team}_home_difficulty.csv', index=False)
    away_df.to_csv(f'{team}_away_difficulty.csv', index=False)

print("Difficulty ratings calculated and saved!")
