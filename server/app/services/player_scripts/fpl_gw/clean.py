import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def clean_fpl_gw_data(fpl_gw_data_dir='fpl_gw_data'):
    """
    Cleans unnecessary columns from FPL gameweek data CSV files based on player positions.

    Parameters:
    fpl_gw_data_dir (str): The directory containing FPL gameweek data. Defaults to 'fpl_gw_data'.
    """
    # Define the columns to drop based on position
    columns_to_drop_by_position = {
        'FWD': ['clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved', 'saves', 'expected_goals_conceded'],
        'MID': ['own_goals', 'penalties_saved', 'saves', 'expected_goals_conceded'],
        'DEF': ['saves', 'penalties_saved'],
        'GK': ['expected_goals', 'goals_scored', 'ict_index', 'penalties_missed']
    }
    # Define columns to always drop
    columns_to_always_drop = [
        'bps', 'creativity', 'element', 'expected_goal_involvements', 'influence', 
        'round', 'team_a_score', 'team_h_score', 'threat', 'transfers_balance', 
        'transfers_in', 'transfers_out', 'value', 'expected_goals', 'goals_scored', 'assists', 
        'expected_assists', 'penalties_missed', 'red_cards', 'yellow_cards', 'fixture', 'xP'
    ]

    # Traverse through each season and each player CSV file
    for season in os.listdir(fpl_gw_data_dir):
        season_dir = os.path.join(fpl_gw_data_dir, season)
        
        if os.path.isdir(season_dir):
            print(f"Cleaning data for season: {season}")
            
            def clean_gameweek_file(gw_file):
                try:
                    file_path = os.path.join(season_dir, gw_file)
                    df = pd.read_csv(file_path)
                    
                    if 'position' in df.columns:
                        first_position = df['position'].iloc[0]  # Assuming the first row has valid data
                        # Drop columns based on position
                        columns_to_drop = columns_to_drop_by_position.get(first_position, [])
                        df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
                        
                        # Drop columns that should always be removed
                        df_cleaned = df_cleaned.drop(columns=[col for col in columns_to_always_drop if col in df_cleaned.columns], errors='ignore')
                        
                        # Map opponent_team ID to team name
                        if 'opponent_team' in df_cleaned.columns:
                            TEAM_IDS = {
                                '2021-22': {1: 'Arsenal', 2: 'Aston Villa', 3: 'Brentford', 4: 'Brighton', 5: 'Burnley',
                                            6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton', 9: 'Leeds United', 10: 'Leicester City',
                                            11: 'Liverpool', 12: 'Manchester City', 13: 'Manchester United', 14: 'Newcastle United', 
                                            15: 'Norwich', 16: 'Southampton', 17: 'Tottenham', 18: 'Watford', 19: 'West Ham', 20: 'Wolverhampton'},
                                '2022-23': {1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford', 5: 'Brighton', 
                                            6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton', 9: 'Fulham', 10: 'Leeds United', 
                                            11: 'Leicester City', 12: 'Liverpool', 13: 'Manchester City', 14: 'Manchester United', 
                                            15: 'Newcastle United', 16: 'Nottingham Forest', 17: 'Southampton', 18: 'Tottenham', 
                                            19: 'West Ham', 20: 'Wolverhampton'},
                                '2023-24': {1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford', 5: 'Brighton', 6: 'Burnley',
                                            7: 'Chelsea', 8: 'Crystal Palace', 9: 'Everton', 10: 'Fulham', 11: 'Liverpool', 
                                            12: 'Luton', 13: 'Manchester City', 14: 'Manchester United', 15: 'Newcastle United', 
                                            16: 'Nottingham Forest', 17: 'Sheffield', 18: 'Tottenham', 19: 'West Ham', 20: 'Wolverhampton'},
                                '2024-25': {1: "Arsenal", 2: "Aston Villa", 3: "Bournemouth", 4: "Brentford", 5: "Brighton",
                                            6: "Chelsea", 7: "Crystal Palace", 8: "Everton", 9: "Fulham", 10: "Ipswich Town",
                                            11: "Leicester City", 12: "Liverpool", 13: "Manchester City", 14: "Manchester United",
                                            15: "Newcastle United", 16: "Nottingham Forest", 17: "Southampton", 18: "Tottenham",
                                            19: "West Ham", 20: "Wolverhampton"}
                            }
                            df_cleaned['opponent_team'] = df_cleaned['opponent_team'].map(TEAM_IDS.get(season, {}))
                        
                        # Modify the kickoff_time column to only show the date, remove rows with invalid dates
                        if 'kickoff_time' in df_cleaned.columns:
                            try:
                                df_cleaned['kickoff_time'] = pd.to_datetime(df_cleaned['kickoff_time'], errors='coerce').dt.date
                                df_cleaned = df_cleaned.dropna(subset=['kickoff_time'])
                            except Exception as e:
                                print(f"Error processing kickoff_time in {gw_file}: {e}")
                        
                        # Save the modified dataframe back to the CSV file
                        df_cleaned.to_csv(file_path, index=False)
                except Exception as e:
                    print(f"Error processing {gw_file}: {e}")
            
            # Use ThreadPoolExecutor to clean gameweek files concurrently
            with ThreadPoolExecutor() as executor:
                gw_files = [gw_file for gw_file in os.listdir(season_dir) if gw_file.endswith('.csv')]
                executor.map(clean_gameweek_file, gw_files)

    print("Finished cleaning all seasons.")


'''
# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    clean_fpl_gw_data()
'''