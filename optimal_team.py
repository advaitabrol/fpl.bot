import os
import pandas as pd
import pulp as pl

# Directory containing the player data
base_dir = './fake_plur_data/'

# Load player data from CSV files and correct column names
def load_player_data(base_dir):
    positions = ['GK', 'DEF', 'MID', 'ATT']
    all_players = pd.DataFrame()

    for pos in positions:
        file_path = os.path.join(base_dir, pos, f"{pos}.csv")
        df = pd.read_csv(file_path)
        
        # Rename columns to match the expected names for consistency
        df.rename(columns={'Player Name': 'name', 'Team': 'team', 'Value': 'value', 'Expected Points': 'expected_points'}, inplace=True)
        
        df['position'] = pos  # Add position column to differentiate
        all_players = pd.concat([all_players, df], ignore_index=True)

    return all_players

# Select the bench players, minimizing their value
def select_bench_players(all_players):
    # Initialize LP problem for selecting bench players with minimum value
    prob = pl.LpProblem("BenchSelection", pl.LpMinimize)

    # Define decision variables for the bench
    player_vars = [pl.LpVariable(f"bench_player_{i}", cat='Binary') for i in range(len(all_players))]

    # Objective: Minimize the total value of the bench
    prob += pl.lpSum([player_vars[i] * all_players.loc[i, 'value'] for i in range(len(all_players))])

    # Constraint: Select exactly 4 bench players
    prob += pl.lpSum(player_vars) == 4

    # Constraint: 1 goalkeeper on the bench
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'GK']) == 1

    # Solve the LP problem
    prob.solve()

    # Get the selected bench players
    bench_players = [i for i in range(len(all_players)) if player_vars[i].varValue == 1]

    return bench_players

# Select the starting 11 players, maximizing expected points
def select_starting_11(all_players, remaining_budget, bench_players):
    # Remove bench players from the pool
    prob = pl.LpProblem("Starting11Selection", pl.LpMaximize)

    # Define decision variables for starting 11
    player_vars = [pl.LpVariable(f"starting_player_{i}", cat='Binary') for i in range(len(all_players))]

    # Objective: Maximize the total expected points for the starting 11
    prob += pl.lpSum([player_vars[i] * all_players.loc[i, 'expected_points'] for i in range(len(all_players))])

    # Constraint: Total value of the starting 11 should be within the remaining budget
    prob += pl.lpSum([player_vars[i] * all_players.loc[i, 'value'] for i in range(len(all_players))]) <= remaining_budget

    # Constraint: Select exactly 11 starting players
    prob += pl.lpSum(player_vars) == 11

    # Position constraints for the starting 11
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'GK']) == 1
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'DEF']) >= 3
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'MID']) >= 3
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.loc[i, 'position'] == 'ATT']) >= 1

    # Solve the LP problem
    prob.solve()

    # Get the selected starting players
    starting_players = [i for i in range(len(all_players)) if player_vars[i].varValue == 1]

    return starting_players

# Main function to build the optimal team
def build_optimal_team():
    all_players = load_player_data(base_dir)

    # Phase 1: Select bench players (minimizing their value)
    bench_players = select_bench_players(all_players)

    # Calculate the total value of the bench players
    total_bench_value = sum(all_players.loc[i, 'value'] for i in bench_players)

    # Remaining budget for the starting 11
    remaining_budget = 100 - total_bench_value

    # Phase 2: Select the starting 11 players (maximizing expected points)
    starting_players = select_starting_11(all_players, remaining_budget, bench_players)

    # Prepare the final team
    team = {'starting': starting_players, 'subs': bench_players}

    return team

# Print the optimal team
def print_team(team, all_players):
    print("Starting 11 players:")
    for i, player_idx in enumerate(team['starting']):
        player = all_players.loc[player_idx]
        print(f"{i+1}. {player['name']} ({player['position']}) - {player['team']} - Value: {player['value']} - Expected Points: {player['expected_points']}")

    print("\nSubstitute players:")
    for i, player_idx in enumerate(team['subs']):
        player = all_players.loc[player_idx]
        print(f"{i+1}. {player['name']} ({player['position']}) - {player['team']} - Value: {player['value']} - Expected Points: {player['expected_points']}")

    # Calculate total values and expected points
    total_value = sum(all_players.loc[player_idx, 'value'] for player_idx in team['starting']) + sum(all_players.loc[player_idx, 'value'] for player_idx in team['subs'])
    total_expected_points = sum(all_players.loc[player_idx, 'expected_points'] for player_idx in team['starting'])

    print(f"\nTotal Value: {total_value}")
    print(f"Total Expected Points (Starting 11 only): {total_expected_points}")

if __name__ == "__main__":
    optimal_team = build_optimal_team()
    all_players = load_player_data(base_dir)
    print_team(optimal_team, all_players)
