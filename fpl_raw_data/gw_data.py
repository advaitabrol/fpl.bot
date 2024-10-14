import pandas as pd


import pandas as pd

def aggregate_player_data(season, gw_ceil, gw_count, player_name):
    """
    Aggregate a player's gameweek data for a specific season up to the specified gameweek.

    :param season: The season to aggregate data for (e.g., '2023-24').
    :param gameweek_limit: The maximum gameweek number to aggregate data for.
    :param player_name: The name of the player to search for.
    :return: A DataFrame containing aggregated stats for the player.
    """
    player_stats = []

    # Iterate through the gameweeks
    for gw in range((gw_ceil - gw_count), gw_ceil):
        try:
            # Load the gameweek data
            df = pd.read_csv(f"gw_data/{season}/gw{gw}.csv")

            # Filter for the player using direct string matching
            player_data = df[df['name'].str.contains(player_name, case=False, na=False)].copy()  # Create a copy

            # If player data is found, append it to the list
            if not player_data.empty:
                player_data['games'] = gw  # Add a column for gameweek
                player_stats.append(player_data)

        except FileNotFoundError:
            print(f"Gameweek data for GW {gw} not found.")
        except Exception as e:
            print(f"An error occurred while processing GW {gw}: {e}")

    # Combine all player stats into a single DataFrame
    if player_stats:
        combined_stats = pd.concat(player_stats, ignore_index=True)

        # Aggregate the data
        aggregated_stats = combined_stats.groupby('name').agg({
            'goals_scored': 'sum',
            'assists': 'sum',
            'bps': 'sum',  # Add any other relevant stats to aggregate
            'games': 'count'  # Count the number of gameweeks the player played
        }).reset_index()

        return aggregated_stats
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no stats found


# Example usage
if __name__ == "__main__":
    season_input = '2023-24'
    gameweek_ceiling = 39  # Specify the number of gameweeks to aggregate
    player_name_input = 'Cole Palmer'  # Specify the player's name
    gameweek_count = 5
    aggregated_player_statistics = aggregate_player_data(season_input, gameweek_ceiling, gameweek_count, player_name_input)

    if not aggregated_player_statistics.empty:
        print(f"Aggregated statistics for {player_name_input} for the last {gameweek_count} game weeks up to GW {gameweek_ceiling}:")
        print(aggregated_player_statistics)
    else:
        print(f"No statistics found for {player_name_input} up to GW {gameweek_ceiling}.")
