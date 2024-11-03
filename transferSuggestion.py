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

def get_team_data(input_names, all_players):
    """Retrieve the data for the input team based on player names."""
    team_data = []
    for name in input_names:
        player_data = all_players[all_players['name'] == name.strip().lower()]
        if player_data.empty:
            raise ValueError(f"Player '{name}' not found in data files.")
        team_data.append(player_data.iloc[0].to_dict())
    return team_data

def validate_team(team):
    """Ensure the team meets constraints such as budget and positional requirements."""
    team_df = pd.DataFrame(team)
    if team_df['price'].sum() > 100:
        return False
    if len(team_df[team_df['position'] == 'GK']) != 2:
        return False
    if len(team_df[team_df['position'] == 'DEF']) != 5:
        return False
    if len(team_df[team_df['position'] == 'MID']) != 5:
        return False
    if len(team_df[team_df['position'] == 'ATT']) != 3:
        return False
    if team_df.groupby('team').size().max() > 3:
        return False
    return True

def calculate_expected_points(team):
    """Calculate total expected points for the top 11 players based on 3-week projection."""
    top_11 = sorted(team, key=lambda x: x['week1'] + x['week2'] + x['week3'], reverse=True)[:11]
    return sum(player['week1'] + player['week2'] + player['week3'] for player in top_11)

def suggest_transfers(input_names, max_transfers, keep=[], blacklist=[]):
    all_players = load_player_data(base_dir)
    current_team = get_team_data(input_names, all_players)
    keep_set = set(name.strip().lower() for name in keep)  # Normalize keep names
    blacklist_set = set(name.strip().lower() for name in blacklist)  # Normalize blacklist names
    transferable_team = [player for player in current_team if player['name'] not in keep_set]

    best_team = current_team.copy()
    best_score = calculate_expected_points(best_team)
    original_team_score = best_score
    transfers_suggestion = []

    # Create a set of names currently in the team for easy lookup
    current_team_names = set(player['name'] for player in current_team)

    for t in range(1, max_transfers + 1):
        for out_players in combinations(transferable_team, t):
            out_budget = sum(player['price'] for player in out_players)
            out_positions = [player['position'] for player in out_players]

            # Filter candidates by position, budget, exclude players in the team and in blacklist
            candidates = all_players[(all_players['position'].isin(out_positions)) & 
                                     (all_players['price'] <= out_budget) & 
                                     (~all_players['name'].isin(current_team_names)) & 
                                     (~all_players['name'].isin(blacklist_set))].copy()
            candidates['total_points'] = candidates[['week1', 'week2', 'week3']].sum(axis=1)
            candidates = candidates.nlargest(15, 'total_points')  # Limit to top 15

            # Match out players by position only
            for in_players in combinations(candidates.to_dict('records'), t):
                if (sum(player['price'] for player in in_players) <= out_budget and
                    all(out['position'] == inp['position'] for out, inp in zip(out_players, in_players))):
                    
                    new_team = [player for player in current_team if player not in out_players] + list(in_players)
                    if validate_team(new_team):
                        new_score = calculate_expected_points(new_team)
                        if new_score > best_score:
                            best_score = new_score
                            best_team = new_team
                            transfers_suggestion = [(out['name'], inp['name']) for out, inp in zip(out_players, in_players)]
    
    before_price = sum(player['price'] for player in current_team)
    after_price = sum(player['price'] for player in best_team)

    return best_team, transfers_suggestion, before_price, after_price, original_team_score

def print_team(best_team, transfers_suggestion, before_price, after_price, original_team_score):
    print("Best Suggested Team:")
    for player in best_team:
        print(f"{player['name'].title()} ({player['position']}) - {player['team']} - Price: {player['price']}")

    print(f"\nTotal Team Price Before Transfers: {before_price}")
    print(f"Total Team Price After Transfers: {after_price}")
    print(f"Total Expected Points of Original Team (Top 11): {original_team_score}")
    total_expected_points = calculate_expected_points(best_team)
    print(f"Total Expected Points After Transfers (Top 11): {total_expected_points}")

    if transfers_suggestion:
        print("\nTransfers Suggestion:")
        for out_name, in_name in transfers_suggestion:
            print(f"Transfer Out: {out_name.title()} -> Transfer In: {in_name.title()}")
    else:
        print("\nNo transfers recommended - save your transfers.")

if __name__ == "__main__":
    input_team = [
        "André Onana", "Leif Davis", "Diogo Dalot Teixeira", 
        "Lucas Digne", "Bryan Mbeumo", "Cole Palmer", "Bukayo Saka", 
        "Mohamed Salah", "Danny Welbeck", "Rasmus Højlund", "Ollie Watkins", 
        "Joe Lumley", "Lewis Hall", "Luke Thomas", "Jakub Moder"
    ]
    max_transfers = 3
    keep = ["Rasmus Højlund"]  # Players to keep in the team
    blacklist = ["Raúl Jiménez", "Nicolas Jackson"]  # Players not to transfer in

    best_team, transfers_suggestion, before_price, after_price, original_team_score = suggest_transfers(input_team, max_transfers, keep, blacklist)
    print_team(best_team, transfers_suggestion, before_price, after_price, original_team_score)
