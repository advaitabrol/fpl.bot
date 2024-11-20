import os
import pandas as pd
from itertools import combinations
from collections import Counter

# Directory containing the player data
base_dir = './prediction_data/2024-25/GW10/'


def load_player_data(base_dir):
    """Load player data for all potential transfers, applying minimum expected points thresholds by position."""
    files = {
        'GK': 'gk.csv',
        'DEF': 'def.csv',
        'MID': 'mid.csv',
        'ATT': 'fwd.csv'
    }
    all_players = pd.DataFrame()
    
    for pos, filename in files.items():
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)
        df['position'] = pos  # Add position column
        df['name'] = df['name'].str.strip().str.lower()  # Normalize player names

        # Calculate total expected points across the 3 weeks
        df['total_expected_points'] = df[['week1', 'week2', 'week3']].sum(axis=1)
        
        # Apply the exclusion criteria based on position
        if pos == 'GK':
            df = df[df['total_expected_points'] >= 7]
        elif pos == 'DEF':
            df = df[df['total_expected_points'] >= 9]
        elif pos == 'MID':
            df = df[df['total_expected_points'] >= 13]
        elif pos == 'ATT':
            df = df[df['total_expected_points'] >= 15]
        
        all_players = pd.concat([all_players, df], ignore_index=True)

    # Ensure required columns are present
    if 'price' not in all_players.columns or 'name' not in all_players.columns:
        raise ValueError("Required columns ('price', 'name') not found in player data files.")
    
    return all_players


def load_input_team(input_team_names, base_dir):
    """Load input team data separately from CSV files based on input team names."""
    files = {
        'GK': 'gk.csv',
        'DEF': 'def.csv',
        'MID': 'mid.csv',
        'ATT': 'fwd.csv'
    }
    input_team = pd.DataFrame()

    for pos, filename in files.items():
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)
        df['position'] = pos
        df['name'] = df['name'].str.strip().str.lower()
        df['total_expected_points'] = df[['week1', 'week2', 'week3']].sum(axis=1)

        # Filter the data to include only players in the input team
        input_team = pd.concat(
            [input_team, df[df['name'].isin([name.lower() for name in input_team_names])]],
            ignore_index=True
        )

    return input_team


def get_position_limits():
    """Defines the constraints for the starting 11 lineup."""
    return {'GK': (1, 1), 'DEF': (3, 5), 'MID': (2, 5), 'ATT': (1, 3)}


def select_optimal_starting_11(team, week, dp_cache):
    """Selects the optimal starting 11 lineup for a given week based on position constraints."""
    team_key = tuple(player['name'] for player in team)
    week_key = f"{team_key}_week{week}"
    
    # Check the cache for the precomputed optimal starting 11 for this team and week
    if week_key in dp_cache:
        return dp_cache[week_key]
    
    # Step 1: Fixed positions selection
    gk_candidates = [player for player in team if player['position'] == 'GK']
    def_candidates = [player for player in team if player['position'] == 'DEF']
    mid_candidates = [player for player in team if player['position'] == 'MID']
    fwd_candidates = [player for player in team if player['position'] == 'ATT']
    
    # Select 1 GK with the most expected points
    selected_gk = max(gk_candidates, key=lambda x: x[f'week{week}'])
    
    # Select 3 DEF, 2 MID, and 1 FWD with the most expected points
    selected_def = sorted(def_candidates, key=lambda x: x[f'week{week}'], reverse=True)[:3]
    selected_mid = sorted(mid_candidates, key=lambda x: x[f'week{week}'], reverse=True)[:2]
    selected_fwd = sorted(fwd_candidates, key=lambda x: x[f'week{week}'], reverse=True)[:1]
    
    # Collect initially selected players
    starting_11 = [selected_gk] + selected_def + selected_mid + selected_fwd
    
    # Step 2: Select the remaining 4 best players
    remaining_candidates = [
        player for player in team
        if player not in starting_11 and player['position'] != 'GK'
    ]
    best_combination = []
    best_combination_points = 0
    
    # Find the combination of 4 remaining players with the highest expected points
    for combo in combinations(remaining_candidates, 4):
        combo_points = sum(player[f'week{week}'] for player in combo)
        if combo_points > best_combination_points:
            best_combination_points = combo_points
            best_combination = combo
    
    # Add the best combination of 4 players to the starting 11
    starting_11.extend(best_combination)
    
    # Ensure exactly 11 players and cache result
    dp_cache[week_key] = starting_11
    return starting_11


def calculate_team_points(team, dp_cache):
    """Calculates the total expected points for the team over 3 weeks, with captain points doubled."""
    total_points = 0
    for week in range(1, 4):
        starting_11 = select_optimal_starting_11(team, week, dp_cache)
        
        # Find the player with the most expected points to designate as the captain
        captain = max(starting_11, key=lambda player: player[f'week{week}'])
        
        # Calculate the total points for the starting 11, doubling the captain's points
        week_points = sum(player[f'week{week}'] for player in starting_11) + captain[f'week{week}']
        
        total_points += week_points
        
    return total_points


def suggest_transfers(input_team_names, max_transfers, keep=[], blacklist=[]):
    # Load all potential transfer players and input team separately
    all_players = load_player_data(base_dir)
    input_team = load_input_team(input_team_names, base_dir)

    # Exclude input team players from the all_players data
    all_players = all_players[~all_players['name'].isin(input_team['name'])]

    keep_set = set(name.strip().lower() for name in keep)
    blacklist_set = set(name.strip().lower() for name in blacklist)
    dp_cache = {}

    # Start with the input team as the initial team
    current_team = input_team.to_dict('records')
    best_team = current_team.copy()
    best_score = calculate_team_points(best_team, dp_cache)
    original_team_score = best_score
    transfers_suggestion = []

    current_team_names = set(player['name'] for player in current_team)

    for t in range(1, max_transfers + 1):
        # Iterate over all combinations of `t` players to transfer out
        for out_players in combinations(current_team, t):
            out_positions = [player['position'] for player in out_players]
            out_points = sum(sum(player[f'week{week}'] for week in range(1, 4)) for player in out_players)
            out_budget = sum(player['price'] for player in out_players)

            # Prepare a list of candidates for each outgoing position
            position_candidates = {}
            for position in out_positions:
                position_candidates[position] = all_players[
                    (all_players['position'] == position) &
                    (~all_players['name'].isin(current_team_names)) &
                    (~all_players['name'].isin(blacklist_set))
                ].to_dict('records')

            # Generate combinations of candidates that match the positions of out_players
            for in_players in combinations(sum(position_candidates.values(), []), t):
                # Ensure in_players match the positions in out_positions exactly
                if sorted(player['position'] for player in in_players) != sorted(out_positions):
                    continue

                # Check if total team budget is under the limit with the new players
                new_team = [player for player in current_team if player not in out_players] + list(in_players)
                total_price = sum(player['price'] for player in new_team)
                
                # Check team constraint: no more than 3 players from the same team
                team_counts = Counter(player['team'] for player in new_team)
                if any(count > 3 for count in team_counts.values()):
                    continue  # Skip this team if it exceeds the 3-player limit per real-world team

                # Ensure there is at least one goalkeeper in new_team
                if total_price <= 100 and any(player['position'] == 'GK' for player in new_team):
                    # Calculate points to see if the new team is better
                    in_points = sum(sum(player[f'week{week}'] for week in range(1, 4)) for player in in_players)
                    if in_points > out_points:
                        new_score = calculate_team_points(new_team, dp_cache)
                        if new_score > best_score:
                            best_score = new_score
                            best_team = new_team
                            transfers_suggestion = [(out['name'], inp['name']) for out, inp in zip(out_players, in_players)]
    
    before_price = sum(player['price'] for player in current_team)
    after_price = sum(player['price'] for player in best_team)

    return best_team, transfers_suggestion, before_price, after_price, original_team_score


def run_test_case():
    input_team = [
        "Aaron Ramsdale", "Ibrahima Konaté", "Joško Gvardiol", 
        "Taylor Harwood-Bellis", "Bryan Mbeumo", "Brennan Johnson", "Bukayo Saka", 
        "Mohamed Salah", "Raúl Jiménez", "Nicolas Jackson", "Ollie Watkins", 
        "Sam Johnstone", "Leif Davis", "Malang Sarr", "Omari Kellyman"
    ]
    max_transfers = 3
    keep = []  # Players to keep in the team
    blacklist = []  # Players not to transfer in

    best_team, transfers_suggestion, before_price, after_price, original_team_score = suggest_transfers(input_team, max_transfers, keep, blacklist)
    print_team_and_suggestion(best_team, transfers_suggestion, before_price, after_price, original_team_score)


def print_team_and_suggestion(best_team, transfers_suggestion, before_price, after_price, original_team_score):
    print("Best Suggested Team:")
    for player in best_team:
        print(f"{player['name'].title()} ({player['position']}) - {player['team']} - Price: {player['price']}")

    print(f"\nTotal Team Price Before Transfers: {before_price}")
    print(f"Total Team Price After Transfers: {after_price}")
    print(f"Total Expected Points of Original Team (Top 11): {original_team_score}")
    total_expected_points = calculate_team_points(best_team, {})
    print(f"Total Expected Points After Transfers (Top 11): {total_expected_points}")

    if transfers_suggestion:
        print("\nTransfers Suggestion:")
        for out_name, in_name in transfers_suggestion:
            print(f"Transfer Out: {out_name.title()} -> Transfer In: {in_name.title()}")
    else:
        print("\nNo transfers recommended - save your transfers.")


if __name__ == "__main__":
    run_test_case()
