import os
import pandas as pd
from fuzzywuzzy import process
from concurrent.futures import ThreadPoolExecutor

def map_team_name(team_name, team_name_mappings):
    """Map team names to their canonical versions."""
    return team_name_mappings.get(team_name, team_name)

def get_closest_match(team_name, options, threshold=40):
    """Get the closest match for a team name using fuzzywuzzy."""
    closest_match, score = process.extractOne(team_name, options)
    return closest_match if score >= threshold else None

def get_difficulty_subdirectory(season, row_index):
    """Determine the appropriate difficulty directory based on the row index."""
    season_start_year = int(season.split('-')[0])
    return f'difficulty_{season_start_year}' if row_index <= 19 else f'difficulty_half_{season_start_year + 1}'

def load_team_file(directory_path, team_name):
    """Load and return the DataFrame for the closest matching team file."""
    if os.path.exists(directory_path):
        team_files = os.listdir(directory_path)
        closest_team_file = get_closest_match(team_name, team_files)
        if closest_team_file:
            return pd.read_csv(os.path.join(directory_path, closest_team_file))
    return pd.DataFrame()

def get_next_fixture_difficulty(player_team, was_home, opponent_team, season, row_index, fixture_difficulties_dir, team_name_mappings):
    """Calculate next week's specific fixture difficulty."""
    mapped_team = map_team_name(player_team, team_name_mappings)
    difficulty_subdir = get_difficulty_subdirectory(season, row_index)
    team_df = load_team_file(os.path.join(fixture_difficulties_dir, difficulty_subdir), mapped_team)

    if not team_df.empty:
        opponent = get_closest_match(opponent_team, team_df['Opponent'].tolist())
        if opponent is not None:
            difficulty = team_df.loc[team_df['Opponent'] == opponent, 'Home Difficulty' if was_home else 'Away Difficulty'].values
            if len(difficulty) > 0:
                return round(difficulty[0], 2)
    return None

def get_next_holistic_fixture_difficulty(opponent_team, was_home, season, row_index, holistic_difficulties_dir, team_name_mappings):
    """Calculate holistic fixture difficulty for the next week."""
    mapped_team = map_team_name(opponent_team, team_name_mappings)
    difficulty_subdir = get_difficulty_subdirectory(season, row_index)
    team_df = load_team_file(os.path.join(holistic_difficulties_dir, difficulty_subdir), mapped_team)

    if not team_df.empty:
        difficulty = team_df['Away Difficulty' if was_home else 'Home Difficulty'].values
        if len(difficulty) > 0:
            return round(difficulty[0], 2)
    return None

def get_next_season_fixture(player_file, current_season_index, position, seasons, merged_data_dir):
    """Handle moving to the next season for the last row."""
    if current_season_index < len(seasons) - 1:
        next_season = seasons[current_season_index + 1]
        next_season_dir = os.path.join(merged_data_dir, next_season, position)
        if os.path.exists(next_season_dir):
            closest_player_file = get_closest_match(player_file, os.listdir(next_season_dir), threshold=75)
            if closest_player_file:
                next_df = pd.read_csv(os.path.join(next_season_dir, closest_player_file))
                if not next_df.empty:
                    return next_df.iloc[0][['team', 'was_home', 'opponent_team']].to_list()
    return None, None, None

def process_player_file(player_file_path, season, season_index, position, seasons, merged_data_dir, fixture_difficulties_dir, holistic_difficulties_dir, fixtures_for_each_team_dir, team_name_mappings):
    """Process a single player file."""
    try:
        df = pd.read_csv(player_file_path)
    except pd.errors.EmptyDataError:
        print(f"File {player_file_path} is empty or corrupted. Skipping.")
        return

    if df.empty or not {'team', 'was_home', 'opponent_team'}.issubset(df.columns):
        return

    df['next_team'] = df['team'].shift(-1)
    df['next_was_home'] = df['was_home'].shift(-1)
    df['next_opponent_team'] = df['opponent_team'].shift(-1)

    if season_index == len(seasons) - 1:
        team_name = map_team_name(df['team'].iloc[-1], team_name_mappings)
        available_files = os.listdir(fixtures_for_each_team_dir)
        matched_file = get_closest_match(f"{team_name}_{season}.csv", available_files)

        if matched_file:
            fixture_df = pd.read_csv(os.path.join(fixtures_for_each_team_dir, matched_file))
            next_fixture_rows = fixture_df[pd.to_numeric(fixture_df['Team Score'], errors='coerce').isna()]

            if not next_fixture_rows.empty:
                # Save data from the first row of next_fixture_rows to the last row of df
                first_row = next_fixture_rows.iloc[0]
                if pd.notnull(first_row.Opponent) and pd.notnull(first_row.Location):
                    df.at[df.index[-1], 'next_team'] = df['team'].iloc[-1]
                    df.at[df.index[-1], 'next_opponent_team'] = first_row.Opponent
                    df.at[df.index[-1], 'next_was_home'] = first_row.Location == 'Home'

                # Create two additional rows with data from next_fixture_rows
                additional_rows = []
                for i, next_fixture_row in enumerate(next_fixture_rows[1:3].itertuples(index=False)):
                    if pd.notnull(next_fixture_row.Opponent) and pd.notnull(next_fixture_row.Location):
                        additional_rows.append({
                            'team': df['team'].iloc[-1],
                            'next_team': df['team'].iloc[-1],
                            'next_opponent_team': next_fixture_row.Opponent,
                            'next_was_home': next_fixture_row.Location == 'Home'
                        })

                if additional_rows:
                    additional_df = pd.DataFrame(additional_rows)
                    df = pd.concat([df, additional_df], ignore_index=True)


    elif season_index < len(seasons) - 1:
        next_season_team, next_season_was_home, next_season_opponent_team = get_next_season_fixture(
            os.path.basename(player_file_path), season_index, position, seasons, merged_data_dir
        )
        if next_season_team:
            df.loc[df.index[-1], ['next_team', 'next_was_home', 'next_opponent_team']] = [
                next_season_team, next_season_was_home, next_season_opponent_team
            ]

    df['next_week_specific_fixture_difficulty'] = df.apply(
        lambda row: get_next_fixture_difficulty(
            row['next_team'], row['next_was_home'], row['next_opponent_team'], season, row.name,
            fixture_difficulties_dir, team_name_mappings
        ) if pd.notnull(row['next_team']) and pd.notnull(row['next_opponent_team']) else None, axis=1
    )

    df['next_week_holistic_fixture_difficulty'] = df.apply(
        lambda row: get_next_holistic_fixture_difficulty(
            row['next_opponent_team'], row['next_was_home'], season, row.name,
            holistic_difficulties_dir, team_name_mappings
        ) if pd.notnull(row['next_opponent_team']) else None, axis=1
    )

    df.to_csv(player_file_path, index=False)
    print(f"Updated {player_file_path} with fixture difficulties.")

def process_fixture_data(merged_data_dir, fixture_difficulties_dir, holistic_difficulties_dir, fixtures_for_each_team_dir, seasons, positions, team_name_mappings):
    """Main function to traverse the data and calculate fixture difficulties."""
    with ThreadPoolExecutor() as executor:
        futures = []
        for season_index, season in enumerate(seasons):
            for position in positions:
                position_dir = os.path.join(merged_data_dir, season, position)

                if not os.path.exists(position_dir):
                    print(f"Position directory {position_dir} does not exist for season {season}.")
                    continue

                for player_file in os.listdir(position_dir):
                    player_file_path = os.path.join(position_dir, player_file)
                    futures.append(executor.submit(
                        process_player_file, player_file_path, season, season_index, position, seasons, merged_data_dir,
                        fixture_difficulties_dir, holistic_difficulties_dir, fixtures_for_each_team_dir, team_name_mappings
                    ))

        for future in futures:
            future.result()
def process_fixtures_default(seasons=['2022-23', '2023-24', '2024-25']):
    """Helper function to call process_form_data with default directories and mappings."""
    merged_data_dir = 'player_data'
    fixture_difficulties_dir = 'fixture_specific_difficulties_incremented'
    holistic_difficulties_dir = 'holistic_difficulties_incremented'
    fixtures_for_each_team_dir = "fixtures_for_each_team"

    positions = ['GK', 'DEF', 'MID', 'FWD']

    team_name_mappings = {
        "Spurs": "Tottenham",
        "Man City": "Manchester City",
        "Man Utd": "Manchester United",
        "Nott'm Forest": "Nottingham Forest"
        "Wolves": "Wolverhampton"
    }

    process_fixture_data(merged_data_dir, fixture_difficulties_dir, holistic_difficulties_dir, fixtures_for_each_team_dir, seasons, positions, team_name_mappings)

if __name__ == "__main__":
    process_fixtures_default()
