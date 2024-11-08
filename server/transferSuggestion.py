import os
import pandas as pd
from itertools import combinations

# Directory containing the player data
base_dir = './prediction_data/2024-25/GW10-12/'

def load_player_data(base_dir):
    """Load player data with 3-week points projections."""
    files = {
        'GK': 'final_gk.csv',
        'DEF': 'final_def.csv',
        'MID': 'final_mid.csv',
        'ATT': 'final_fwd.csv'
    }
    all_players = pd.DataFrame()
    
    for pos, filename in files.items():
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)
        df['position'] = pos  # Add position column
        df['name'] = df['name'].str.strip().str.lower()  # Normalize player names
        all_players = pd.concat([all_players, df], ignore_index=True)
    
    if 'price' not in all_players.columns or 'name' not in all_players.columns:
        raise ValueError("Required columns ('price', 'name') not found in player data files.")
    
    return all_players

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
    
    limits = get_position_limits()
    starting_11 = []
    
    # Position-based selection
    for position, (min_required, max_allowed) in limits.items():
        players = sorted(
            [player for player in team if player['position'] == position],
            key=lambda x: x[f'week{week}'],
            reverse=True
        )
        starting_11.extend(players[:min_required])
        remaining_players = players[min_required:max_allowed]
        starting_11 += remaining_players
    
    # Fill remaining slots with best-performing players
    remaining_slots = 11 - len(starting_11)
    other_positions = [player for player in team if player not in starting_11]
    other_positions_sorted = sorted(other_positions, key=lambda x: x[f'week{week}'], reverse=True)
    starting_11 += other_positions_sorted[:remaining_slots]
    
    # Ensure exactly 11 players and cache result
    starting_11 = starting_11[:11]
    dp_cache[week_key] = starting_11
    return starting_11

def calculate_team_points(team, dp_cache):
    """Calculates the total expected points for the team over 3 weeks."""
    total_points = 0
    for week in range(1, 4):
        starting_11 = select_optimal_starting_11(team, week, dp_cache)
        total_points += sum(player[f'week{week}'] for player in starting_11)
    return total_points

def suggest_transfers(input_names, max_transfers, keep=[], blacklist=[]):
    all_players = load_player_data(base_dir)
    current_team = [player for player in all_players.to_dict('records') if player['name'] in [name.lower() for name in input_names]]
    keep_set = set(name.strip().lower() for name in keep)
    blacklist_set = set(name.strip().lower() for name in blacklist)
    dp_cache = {}

    best_team = current_team.copy()
    best_score = calculate_team_points(best_team, dp_cache)
    original_team_score = best_score
    transfers_suggestion = []

    current_team_names = set(player['name'] for player in current_team)

    for t in range(1, max_transfers + 1):
        # Iterate over all combinations of `t` players to transfer out
        for out_players in combinations(current_team, t):
            out_budget = sum(player['price'] for player in out_players)
            out_positions = [player['position'] for player in out_players]
            out_points = sum(sum(player[f'week{week}'] for week in range(1, 4)) for player in out_players)

            # Filter candidates based on position and price constraints
            for out_player in out_players:
                candidates = all_players[
                    (all_players['position'] == out_player['position']) &
                    (all_players['price'] <= out_budget) &
                    (~all_players['name'].isin(current_team_names)) &
                    (~all_players['name'].isin(blacklist_set))
                ]
                
                candidates = candidates.to_dict('records')

                # Generate combinations of candidates for multiple transfers
                for in_players in combinations(candidates, t):
                    if sum(player['price'] for player in in_players) <= out_budget:
                        # Ensure that the combined points of the in_players are higher than out_players
                        in_points = sum(sum(player[f'week{week}'] for week in range(1, 4)) for player in in_players)
                        if in_points > out_points:
                            new_team = [player for player in current_team if player not in out_players] + list(in_players)
                            new_score = calculate_team_points(new_team, dp_cache)

                            if new_score > best_score:
                                best_score = new_score
                                best_team = new_team
                                transfers_suggestion = [(out['name'], inp['name']) for out, inp in zip(out_players, in_players)]
    
    before_price = sum(player['price'] for player in current_team)
    after_price = sum(player['price'] for player in best_team)

    return best_team, transfers_suggestion, before_price, after_price, original_team_score

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

def run_test_case():
    input_team = [
        "Aaron Ramsdale", "Leif Davis", "Diogo Dalot Teixeira", 
        "Lucas Digne", "Bryan Mbeumo", "Cole Palmer", "Bukayo Saka", 
        "Mohamed Salah", "Danny Welbeck", "Rasmus Højlund", "Ollie Watkins", 
        "Joe Lumley", "Lewis Hall", "Luke Thomas", "Jakub Moder"
    ]
    max_transfers = 3
    keep = []  # Players to keep in the team
    blacklist = []  # Players not to transfer in

    best_team, transfers_suggestion, before_price, after_price, original_team_score = suggest_transfers(input_team, max_transfers, keep, blacklist)
    print_team_and_suggestion(best_team, transfers_suggestion, before_price, after_price, original_team_score)

if __name__ == "__main__":
    run_test_case()
