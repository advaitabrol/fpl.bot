import os
import pandas as pd

# Define the seasons for each calculation
season_sets = {
    "difficulty_2022": ["2019-20", "2020-21", "2021-22"],
    "difficulty_half_2023": ["2019-20", "2020-21", "2021-22", "2022-23_first_half"],
    "difficulty_2023": ["2019-20", "2020-21", "2021-22", "2022-23"],
    "difficulty_half_2024": ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24_first_half"],
    "difficulty_2024": ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
}

# Directory for the FPL team data and output
fpl_team_data_dir = './fpl_team_data/'
base_output_dir = './holistic_difficulties/'

# Helper function to trim the first 19 rows for partial seasons
def load_filtered_data(season, team_name):
    file_path = os.path.join(fpl_team_data_dir, f"{team_name}_{season}.csv")
    
    if season.endswith("first_half"):
        # Handle partial seasons: only take the first 19 rows
        df = pd.read_csv(file_path).head(19)
    else:
        # Load the full season
        df = pd.read_csv(file_path)
    
    return df

# Function to normalize team names (converting underscores to spaces for display, but keeping underscores for file operations)
def display_team_name(filename):
    return filename.replace('_', ' ')  # Convert underscores to spaces for display

# Function to calculate holistic difficulties
def calculate_holistic_difficulties(fpl_team_data_dir, output_dir, season_set):
    teams = set('_'.join(filename.split('_')[:-1]) for filename in os.listdir(fpl_team_data_dir) if filename.endswith('.csv'))
    
    # Loop through each team
    for team in teams:
        total_home_difficulty = 0
        total_away_difficulty = 0
        total_home_matches = 0
        total_away_matches = 0
        
        # Loop through the specified seasons and accumulate difficulty
        for season in season_set:
            try:
                # Load the team's data for that season
                df = load_filtered_data(season, team)
                
                # Separate home and away games
                home_games = df[df['h_a'] == 'h']
                away_games = df[df['h_a'] == 'a']
                
                # Calculate difficulty (weights applied)
                home_difficulty = ((home_games['missed_normalized'] - home_games['scored_normalized']) * 5 +
                                   (home_games['xGA_normalized'] - home_games['xG_normalized']) * 3 +
                                   (home_games['ppda_allowed_normalized'] - home_games['ppda_normalized']) * 2 +
                                   (home_games['deep_normalized'] - home_games['deep_allowed_normalized']) * 2).sum()
                away_difficulty = ((away_games['missed_normalized'] - away_games['scored_normalized']) * 5 +
                                   (away_games['xGA_normalized'] - away_games['xG_normalized']) * 3 +
                                   (away_games['ppda_allowed_normalized'] - away_games['ppda_normalized']) * 2 +
                                   (away_games['deep_normalized'] - away_games['deep_allowed_normalized']) * 2).sum()

                # Update totals and match counts
                total_home_difficulty += home_difficulty
                total_away_difficulty += away_difficulty
                total_home_matches += len(home_games)
                total_away_matches += len(away_games)
            except FileNotFoundError:
                print(f"Data for {team} in {season} not found, skipping this season.")
        
        # Calculate average difficulty across seasons
        avg_home_difficulty = total_home_difficulty / total_home_matches if total_home_matches > 0 else None
        avg_away_difficulty = total_away_difficulty / total_away_matches if total_away_matches > 0 else None

        # Prepare and save the output
        output_path = os.path.join(output_dir, f"{team}_difficulty.csv")
        with open(output_path, 'a') as f:
            if os.stat(output_path).st_size == 0:
                # If file is empty, write header row
                f.write("Team,Home Difficulty,Away Difficulty\n")
            f.write(f"{display_team_name(team)},{avg_home_difficulty},{avg_away_difficulty}\n")
        
        print(f"Saved difficulties for {team} in {output_path}")

# Main function to loop over all season sets and calculate difficulties
def main():
    for folder_name, season_set in season_sets.items():
        # Create output directory
        output_dir = os.path.join(base_output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate holistic difficulties for each season set
        calculate_holistic_difficulties(fpl_team_data_dir, output_dir, season_set)

if __name__ == "__main__":
    main()
