import os
import pandas as pd
import pulp as pl
import numpy as np

# Directory containing the player data

base_dir = './prediction_data/2024-25/GW12/'

def load_player_data(base_dir):
    """Load player data with 3-week points projections."""
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
        all_players = pd.concat([all_players, df], ignore_index=True)
    
    return all_players

def build_wildcard_team(base_dir):
    all_players = load_player_data(base_dir)
    prob = pl.LpProblem("TeamSelection", pl.LpMaximize)

    player_vars = [pl.LpVariable(f"player_{i}", cat='Binary') for i in range(len(all_players))]
    start_vars = [[pl.LpVariable(f"start_week{w}_{i}", cat='Binary') for i in range(len(all_players))] for w in range(3)]

    total_expected_points = sum(
        pl.lpSum([start_vars[w][i] * all_players.iloc[i][f'week{w+1}'] for i in range(len(all_players))])
        for w in range(3)
    )
    prob += total_expected_points

    # Constraints for total team and starting lineup
    prob += pl.lpSum(player_vars) == 15
    for w in range(3):
        prob += pl.lpSum(start_vars[w]) == 11

    # Link starting team with selected team
    for i in range(len(all_players)):
        for w in range(3):
            prob += start_vars[w][i] <= player_vars[i]
            
    prob += pl.lpSum([player_vars[i] * all_players.iloc[i]['price'] for i in range(len(all_players))]) <= 100

    # Position constraints for total team
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'GK']) == 2
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'DEF']) == 5
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'MID']) == 5
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'ATT']) == 3

    # Position constraints for each week’s starting lineup
    for w in range(3):
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'GK']) == 1
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'DEF']) >= 3
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'MID']) >= 2
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'ATT']) >= 1

    prob.solve()

    selected_players = [i for i in range(len(all_players)) if player_vars[i].varValue == 1]
    if not selected_players:
        return {"team": [], "error": "No players selected for the team"}

    starting_players = [[i for i in range(len(all_players)) if start_vars[w][i].varValue == 1] for w in range(3)]

    # Prepare JSON structure
    team_json = {"team": [], "captains": []}

    # Determine captains for each week
    captains = []
    for w in range(3):
        if starting_players[w]:
            captain_index = max(
                starting_players[w],
                key=lambda idx: all_players.iloc[idx][f'week{w+1}']
            )
            captains.append(captain_index)
        else:
            captains.append(None)

    for i in selected_players:
        player_data = all_players.iloc[i]
        expected_points = [
            float(player_data[f'week{w+1}']) * (2 if captains[w] is not None and captains[w] == i else 1)
            for w in range(3)
        ]
        is_bench = [bool(i not in starting_players[w]) for w in range(3)]
        is_captain = [bool(captains[w] == i) for w in range(3)]

        team_json["team"].append({
            "name": player_data['name'],
            "team": player_data['team'],
            "position": player_data['position'],
            "price": float(player_data['price']),
            "expected_points": expected_points,
            "isBench": is_bench,
            "isCaptain": is_captain, 
        })

        team_json["captains"] = captains

    return team_json



def print_team(starting_players, bench_players, all_players):
    for w, (week_starting, week_bench) in enumerate(zip(starting_players, bench_players), 1):
        print(f"Starting 11 players for Week {w}:")
        week_players = all_players.iloc[week_starting]
        captain_idx = week_players['week' + str(w)].idxmax()
        captain = week_players.loc[captain_idx]

        week_points = 0  # Initialize weekly expected points
        for i, player_idx in enumerate(week_starting):
            player = all_players.iloc[player_idx]
            is_captain = player_idx == captain_idx
            player_points = player[f'week{w}'] * (2 if is_captain else 1)
            week_points += player_points  # Add to weekly points total
            captain_status = "(C)" if is_captain else ""
            print(f"{i+1}. {player['name']} {captain_status} ({player['position']}) - {player['team']} - "
                  f"Price: {player['price']} - Expected Points: {player_points}")

        print(f"\nTotal Expected Points for Week {w}: {week_points}\n")

        print(f"\nBench players for Week {w}:")
        for player_idx in week_bench:
            player = all_players.iloc[player_idx]
            print(f"- {player['name']} ({player['position']}) - {player['team']} - Price: {player['price']}")

        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    all_players = load_player_data(base_dir)
    starting_players, bench_players = build_optimal_team()
    print_team(starting_players, bench_players, all_players)
