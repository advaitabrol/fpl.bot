import os
import pandas as pd
import pulp as pl

# Directory containing the player data
base_dir = './prediction_data/2024-25/GW9/'

# Load player data from CSV files
def load_player_data(base_dir):
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
        df['position'] = pos  # Add position column to differentiate
        all_players = pd.concat([all_players, df], ignore_index=True)

    return all_players

# Main function to build the optimal team
def build_optimal_team():
    all_players = load_player_data(base_dir)
    
    # Initialize LP problem for selecting 15 players
    prob = pl.LpProblem("TeamSelection", pl.LpMaximize)

    # Define decision variables for each player (1 if selected, 0 otherwise)
    player_vars = [pl.LpVariable(f"player_{i}", cat='Binary') for i in range(len(all_players))]

    # Define a binary variable for whether the player is in the starting 11 (1) or not (0)
    start_vars = [pl.LpVariable(f"start_{i}", cat='Binary') for i in range(len(all_players))]

    # Objective: Maximize the total expected points for the starting 11
    prob += pl.lpSum([start_vars[i] * all_players.loc[i, 'predicted_next_week_points'] for i in range(len(all_players))])

    # Constraint: Select exactly 15 players (starting 11 + 4 substitutes)
    prob += pl.lpSum(player_vars) == 15

    # Starting 11 must include exactly 11 players
    prob += pl.lpSum(start_vars) == 11

    # Link starting players to the selected players (if a player is selected for the starting 11, they must be in the overall team)
    for i in range(len(all_players)):
        prob += start_vars[i] <= player_vars[i]

    # Total value constraint: The team cannot exceed 100 in price
    prob += pl.lpSum([player_vars[i] * all_players.loc[i, 'price'] for i in range(len(all_players))]) <= 100

    # Position constraints for the entire team (15 players)
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'GK']) == 2
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'DEF']) == 5
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'MID']) == 5
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'ATT']) == 3

    # Starting 11 position constraints
    prob += pl.lpSum([start_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'GK']) == 1
    prob += pl.lpSum([start_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'DEF']) >= 3
    prob += pl.lpSum([start_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'MID']) >= 2
    prob += pl.lpSum([start_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'ATT']) >= 1

    # Solve the LP problem
    prob.solve()

    # Get the selected players (starting 11 and bench)
    selected_players = [i for i in range(len(all_players)) if player_vars[i].varValue == 1]
    starting_players = [i for i in range(len(all_players)) if start_vars[i].varValue == 1]
    bench_players = [i for i in selected_players if i not in starting_players]

    return starting_players, bench_players

# Print the optimal team
def print_team(starting_players, bench_players, all_players):
    print("Starting 11 players:")
    for i, player_idx in enumerate(starting_players):
        player = all_players.loc[player_idx]
        print(f"{i+1}. {player['name']} ({player['position']}) - {player['team']} - Price: {player['price']} - Expected Points: {player['predicted_next_week_points']}")

    print("\nSubstitute players:")
    for i, player_idx in enumerate(bench_players):
        player = all_players.loc[player_idx]
        print(f"{i+1}. {player['name']} ({player['position']}) - {player['team']} - Price: {player['price']} - Expected Points: {player['predicted_next_week_points']}")

    # Calculate total value and expected points
    total_value = sum(all_players.loc[player_idx, 'price'] for player_idx in starting_players + bench_players)
    total_expected_points = sum(all_players.loc[player_idx, 'predicted_next_week_points'] for player_idx in starting_players)

    print(f"\nTotal Value: {total_value}")
    print(f"Total Expected Points (Starting 11 only): {total_expected_points}")

if __name__ == "__main__":
    all_players = load_player_data(base_dir)
    starting_players, bench_players = build_optimal_team()
    print_team(starting_players, bench_players, all_players)
