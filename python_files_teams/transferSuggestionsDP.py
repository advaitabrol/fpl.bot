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

def dp_team_selection(all_players, budget=100):
    all_players['total_points'] = all_players[['week1', 'week2', 'week3']].sum(axis=1)
    all_players = all_players.sort_values(by='total_points', ascending=False).reset_index(drop=True)

    num_players = len(all_players)
    dp = [[[-1 for _ in range(budget + 1)] for _ in range(15)] for _ in range(num_players + 1)]
    dp[0][0][0] = 0

    for i in range(1, num_players + 1):
        player = all_players.iloc[i - 1]
        price = int(player['price'])
        points = player['total_points']
        position = player['position']

        for j in range(15):
            for b in range(budget + 1):
                dp[i][j][b] = dp[i - 1][j][b]
                if b >= price and j > 0 and dp[i - 1][j - 1][b - price] != -1:
                    if dp[i][j][b] == -1 or dp[i][j][b] < dp[i - 1][j - 1][b - price] + points:
                        dp[i][j][b] = dp[i - 1][j - 1][b - price] + points

    best_points, best_budget, best_team = 0, 0, []

    for b in range(budget + 1):
        if dp[num_players][11][b] > best_points:
            best_points = dp[num_players][11][b]
            best_budget = b

    i, j, b = num_players, 11, best_budget
    while i > 0 and j > 0 and b > 0:
        if dp[i][j][b] != dp[i - 1][j][b]:
            player = all_players.iloc[i - 1]
            best_team.append(player)
            j -= 1
            b -= int(player['price'])
        i -= 1

    best_team_df = pd.DataFrame(best_team)
    total_points = best_team_df['total_points'].sum() if not best_team_df.empty else 0
    total_price = best_team_df['price'].sum() if not best_team_df.empty else 0

    return best_team_df, total_points, total_price

def print_team(best_team, total_points, total_price):
    print("Best Suggested Team:")
    for index, player in best_team.iterrows():
        print(f"{player['name'].title()} ({player['position']}) - {player['team']} - Price: {player['price']}")

    print(f"\nTotal Team Price: {total_price}")
    print(f"Total Expected Points: {total_points}")

if __name__ == "__main__":
    all_players = load_player_data(base_dir)
    best_team, total_points, total_price = dp_team_selection(all_players)
    print_team(best_team, total_points, total_price)
