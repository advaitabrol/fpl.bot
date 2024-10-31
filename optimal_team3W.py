import os
import pandas as pd
import pulp as pl

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
        all_players = pd.concat([all_players, df], ignore_index=True)
    
    return all_players

def build_optimal_team():
    all_players = load_player_data(base_dir)
    prob = pl.LpProblem("TeamSelection", pl.LpMaximize)

    player_vars = [pl.LpVariable(f"player_{i}", cat='Binary') for i in range(len(all_players))]
    start_vars = [[pl.LpVariable(f"start_week{w}_{i}", cat='Binary') for i in range(len(all_players))] for w in range(3)]

    total_expected_points = sum(
        pl.lpSum([start_vars[w][i] * all_players.iloc[i][f'week{w+1}'] for i in range(len(all_players))])
        for w in range(3)
    )
    prob += total_expected_points

    prob += pl.lpSum(player_vars) == 15
    for w in range(3):
        prob += pl.lpSum(start_vars[w]) == 11

    for i in range(len(all_players)):
        for w in range(3):
            prob += start_vars[w][i] <= player_vars[i]
            
    prob += pl.lpSum([player_vars[i] * all_players.iloc[i]['price'] for i in range(len(all_players))]) <= 100

    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'GK']) == 2
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'DEF']) == 5
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'MID']) == 5
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'ATT']) == 3

    for w in range(3):
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'GK']) == 1
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'DEF']) >= 3
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'MID']) >= 2
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'ATT']) >= 1

    prob.solve()

    selected_players = [i for i in range(len(all_players)) if player_vars[i].varValue == 1]
    starting_players = [[i for i in range(len(all_players)) if start_vars[w][i].varValue == 1] for w in range(3)]
    bench_players = [i for i in selected_players if i not in set(sum(starting_players, []))]

    return starting_players, bench_players

def print_team(starting_players, bench_players, all_players):
    for w, week_starting in enumerate(starting_players, 1):
        print(f"Starting 11 players for Week {w}:")
        week_players = all_players.iloc[week_starting]
        captain_idx = week_players['week' + str(w)].idxmax()
        captain = week_players.loc[captain_idx]

        for i, player_idx in enumerate(week_starting):
            player = all_players.iloc[player_idx]
            captain_status = "(C)" if player_idx == captain_idx else ""
            print(f"{i+1}. {player['name']} {captain_status} ({player['position']}) - {player['team']} - "
                  f"Price: {player['price']} - Expected Points: {player['week' + str(w)] * (2 if player_idx == captain_idx else 1)}")

    print("\nBench players:")
    for player_idx in bench_players:
        player = all_players.iloc[player_idx]
        print(f"- {player['name']} ({player['position']}) - {player['team']} - Price: {player['price']}")

if __name__ == "__main__":
    all_players = load_player_data(base_dir)
    starting_players, bench_players = build_optimal_team()
    print_team(starting_players, bench_players, all_players)
import os
import pandas as pd
import pulp as pl

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
        all_players = pd.concat([all_players, df], ignore_index=True)
    
    return all_players

def build_optimal_team():
    all_players = load_player_data(base_dir)
    prob = pl.LpProblem("TeamSelection", pl.LpMaximize)

    player_vars = [pl.LpVariable(f"player_{i}", cat='Binary') for i in range(len(all_players))]
    start_vars = [[pl.LpVariable(f"start_week{w}_{i}", cat='Binary') for i in range(len(all_players))] for w in range(3)]

    total_expected_points = sum(
        pl.lpSum([start_vars[w][i] * all_players.iloc[i][f'week{w+1}'] for i in range(len(all_players))])
        for w in range(3)
    )
    prob += total_expected_points

    prob += pl.lpSum(player_vars) == 15
    for w in range(3):
        prob += pl.lpSum(start_vars[w]) == 11

    for i in range(len(all_players)):
        for w in range(3):
            prob += start_vars[w][i] <= player_vars[i]
            
    prob += pl.lpSum([player_vars[i] * all_players.iloc[i]['price'] for i in range(len(all_players))]) <= 100

    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'GK']) == 2
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'DEF']) == 5
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'MID']) == 5
    prob += pl.lpSum([player_vars[i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'ATT']) == 3

    for w in range(3):
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'GK']) == 1
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'DEF']) >= 3
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'MID']) >= 2
        prob += pl.lpSum([start_vars[w][i] for i in range(len(all_players)) if all_players.iloc[i]['position'] == 'ATT']) >= 1

    prob.solve()

    selected_players = [i for i in range(len(all_players)) if player_vars[i].varValue == 1]
    starting_players = [[i for i in range(len(all_players)) if start_vars[w][i].varValue == 1] for w in range(3)]
    bench_players = [i for i in selected_players if i not in set(sum(starting_players, []))]

    return starting_players, bench_players

def print_team(starting_players, bench_players, all_players):
    for w, week_starting in enumerate(starting_players, 1):
        print(f"Starting 11 players for Week {w}:")
        week_players = all_players.iloc[week_starting]
        captain_idx = week_players['week' + str(w)].idxmax()
        captain = week_players.loc[captain_idx]

        for i, player_idx in enumerate(week_starting):
            player = all_players.iloc[player_idx]
            captain_status = "(C)" if player_idx == captain_idx else ""
            print(f"{i+1}. {player['name']} {captain_status} ({player['position']}) - {player['team']} - "
                  f"Price: {player['price']} - Expected Points: {player['week' + str(w)] * (2 if player_idx == captain_idx else 1)}")

    print("\nBench players:")
    for player_idx in bench_players:
        player = all_players.iloc[player_idx]
        print(f"- {player['name']} ({player['position']}) - {player['team']} - Price: {player['price']}")

if __name__ == "__main__":
    all_players = load_player_data(base_dir)
    starting_players, bench_players = build_optimal_team()
    print_team(starting_players, bench_players, all_players)
