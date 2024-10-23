import pulp as pl
import pandas as pd
import os

# Load player data for all positions (GK, DEF, MID, ATT) from respective CSVs
def load_player_data(base_dir):
    positions = ['GK', 'DEF', 'MID', 'ATT']
    all_players = []

    for pos in positions:
        file_path = os.path.join(base_dir, f"{pos}.csv")
        df = pd.read_csv(file_path)
        df['Position'] = pos  # Add the position column
        all_players.extend(df.to_dict(orient='records'))
    
    return all_players

# Define the MILP optimization function
def build_optimal_team(player_data, max_team_budget=100, max_players_per_team=3):
    # Create a MILP problem to maximize expected points
    prob = pl.LpProblem("FPL_Team_Selection", pl.LpMaximize)

    # Define decision variables: 1 if the player is selected, 0 otherwise
    player_vars = {player['Name']: pl.LpVariable(player['Name'], cat='Binary') for player in player_data}

    # Objective: Maximize the total expected points of the selected players
    prob += pl.lpSum([player_vars[player['Name']] * player['Expected Points'] for player in player_data])

    # Constraint 1: Total team value must not exceed 100
    prob += pl.lpSum([player_vars[player['Name']] * player['Value'] for player in player_data]) <= max_team_budget, "TotalValue"

    # Constraint 2: 2 goalkeepers, 5 defenders, 5 midfielders, 3 attackers
    prob += pl.lpSum([player_vars[player['Name']] for player in player_data if player['Position'] == 'GK']) == 2, "Goalkeepers"
    prob += pl.lpSum([player_vars[player['Name']] for player in player_data if player['Position'] == 'DEF']) == 5, "Defenders"
    prob += pl.lpSum([player_vars[player['Name']] for player in player_data if player['Position'] == 'MID']) == 5, "Midfielders"
    prob += pl.lpSum([player_vars[player['Name']] for player in player_data if player['Position'] == 'ATT']) == 3, "Attackers"

    # Constraint 3: Only one substitute GK (1 additional goalkeeper must be on the bench)
    prob += pl.lpSum([player_vars[player['Name']] for player in player_data if player['Position'] == 'GK']) == 2, "SubGoalkeeper"

    # Constraint 4: No more than 3 players from the same team
    teams = list(set([player['Team'] for player in player_data]))
    for team in teams:
        prob += pl.lpSum([player_vars[player['Name']] for player in player_data if player['Team'] == team]) <= max_players_per_team, f"MaxPlayersFrom_{team}"

    # Solve the problem
    prob.solve()

    # Extract the selected players
    selected_players = [player for player in player_data if player_vars[player['Name']].varValue == 1]
    
    # Separate into starters and subs based on positions and optimize the lineup
    starters = []
    subs = []
    
    # Sort selected players into starters and subs
    gks = [p for p in selected_players if p['Position'] == 'GK']
    defs = [p for p in selected_players if p['Position'] == 'DEF']
    mids = [p for p in selected_players if p['Position'] == 'MID']
    atts = [p for p in selected_players if p['Position'] == 'ATT']

    # Start with selecting the starters
    starters += gks[:1]  # 1 starting GK
    starters += defs[:5]  # 5 starting DEF
    starters += mids[:5]  # 5 starting MID
    starters += atts[:3]  # 3 starting ATT

    # Now, select substitutes
    subs += gks[1:2]  # 1 sub GK
    subs += defs[5:]  # Remaining defenders as subs
    subs += mids[5:]  # Remaining midfielders as subs
    subs += atts[3:]  # Remaining attackers as subs

    return starters, subs

# Load player data
base_dir = "./fake_plur_data"  # Directory where your player data is stored
players = load_player_data(base_dir)

# Build the optimal team
starters, subs = build_optimal_team(players)

# Display the selected team
print("\n--- Starters ---")
for player in starters:
    print(f"{player['Name']} ({player['Position']}, {player['Team']}) - Value: {player['Value']}, Expected Points: {player['Expected Points']}")

print("\n--- Substitutes ---")
for player in subs:
    print(f"{player['Name']} ({player['Position']}, {player['Team']}) - Value: {player['Value']}, Expected Points: {player['Expected Points']}")
