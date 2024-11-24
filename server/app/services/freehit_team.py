import os
import pandas as pd
import pulp as pl

def load_player_data(base_dir):
    """Load player data from CSV files for a given game week directory."""
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


def build_freehit_team(base_dir):
    """Build the optimal free-hit team for the next game week."""
    all_players = load_player_data(base_dir)
    prob = pl.LpProblem("TeamSelection", pl.LpMaximize)

    # Decision variables
    player_vars = [pl.LpVariable(f"player_{i}", cat='Binary') for i in range(len(all_players))]
    start_vars = [pl.LpVariable(f"start_{i}", cat='Binary') for i in range(len(all_players))]

    # Objective: Maximize points for the next week (week 1)
    prob += pl.lpSum([start_vars[i] * all_players.iloc[i]['week1'] for i in range(len(all_players))])

    # Constraints for total team and starting lineup
    prob += pl.lpSum(player_vars) == 15  # Total of 15 players
    prob += pl.lpSum(start_vars) == 11  # Starting XI players

    # Link starting players to selected team
    for i in range(len(all_players)):
        prob += start_vars[i] <= player_vars[i]

    # Budget constraint
    prob += pl.lpSum([player_vars[i] * all_players.iloc[i]['price'] for i in range(len(all_players))]) <= 100

    # Position constraints for total team
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'GK']) == 2
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'DEF']) == 5
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'MID']) == 5
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'ATT']) == 3

    # Position constraints for the starting XI
    prob += pl.lpSum([start_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'GK']) == 1
    prob += pl.lpSum([start_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'DEF']) >= 3
    prob += pl.lpSum([start_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'MID']) >= 2
    prob += pl.lpSum([start_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'ATT']) >= 1

    # Solve the optimization problem
    prob.solve()

    # Selected players
    selected_players = [i for i in range(len(all_players)) if player_vars[i].varValue == 1]
    if not selected_players:
        return {"team": [], "error": "No players selected for the team"}

    starting_players = [i for i in range(len(all_players)) if start_vars[i].varValue == 1]
    bench_players = [i for i in selected_players if i not in starting_players]

    # Determine captain (highest expected points in starting XI for the week)
    captain_idx = None
    if starting_players:
        captain_idx = max(
            starting_players,
            key=lambda idx: all_players.iloc[idx]['week1']
        )

    # Prepare JSON structure
    team_json = {"team": []}

    for i in selected_players:
        player_data = all_players.iloc[i]
        expected_points = float(player_data['week1']) * (2 if captain_idx == i else 1)
        is_bench = bool(i in bench_players)
        is_captain = bool(captain_idx == i)

        team_json["team"].append({
            "name": player_data['name'],
            "team": player_data['team'],
            "position": player_data['position'],
            "price": float(player_data['price']),
            "expected_points": [expected_points],
            "isBench": [is_bench],
            "isCaptain": is_captain,
        })

    return team_json


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
