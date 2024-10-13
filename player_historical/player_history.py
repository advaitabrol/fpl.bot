import pandas as pd

### this method gets all the data for a specific season, and filters by a field
def load_and_sort_data(season, field, ascending=False):
    """
    Load player data for the specified season and sort by the given field.

    :param season: The season to load data for (e.g., '2023_24').
    :param field: The field to sort by (e.g., 'assists').
    :param ascending: Sort order, False for descending (default), True for ascending.
    :return: A sorted DataFrame.
    """
    # Load the data
    df = pd.read_csv(f"player_historical/player_data_{season}.csv")
    
    # Check if the field exists in the DataFrame
    if field not in df.columns:
        raise ValueError(f"Field '{field}' does not exist in the DataFrame.")

    # Sort the DataFrame by the specified field
    return df.sort_values(by=field, ascending=ascending)


def search_for_player(player_name, seasons=['2021_22', '2022_23', '2023_24']):
    """
    Search for a player across multiple seasons and compile their stats into a DataFrame.

    :param player_name: The name of the player to search for.
    :param seasons: List of seasons to search through (default: ['2021_22', '2022_23', '2023_24']).
    :return: A DataFrame containing the player's stats from each season, sorted by season.
    """
    player_stats = []

    for season in seasons:
        try:
            # Load the data for the season
            df = pd.read_csv(f"player_historical/player_data_{season}.csv")

            # Filter the DataFrame for the specified player
            player_data = df[df['Name'].str.contains(rf'^{player_name}(_\w+)?$', case=False, na=False)]

            # Add season information to the player's stats
            if not player_data.empty:
                player_data['season'] = season
                player_stats.append(player_data)

        except FileNotFoundError:
            print(f"Data for season {season} not found.")
        except Exception as e:
            print(f"An error occurred while processing {season}: {e}")

    # Combine all player stats into a single DataFrame
    if player_stats:
        combined_stats = pd.concat(player_stats, ignore_index=True)

        # Sort the combined DataFrame by season in descending order
        combined_stats['season'] = pd.Categorical(combined_stats['season'], categories=seasons, ordered=True)
        combined_stats = combined_stats.sort_values(by='season', ascending=False)

        return combined_stats
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no stats found

# Example usage
if __name__ == "__main__":
    player_name_input = 'Cole_Palmer'  # Specify the player's name
    player_statistics = search_for_player(player_name_input)
    season_input = '2023_24'  # Specify the season
    field_input = 'assists'     # Specify the field to sort by

    try:
        sorted_data = load_and_sort_data(season_input, field_input, ascending=False)
        print(f"Sorted by '{field_input}' for season '{season_input}':")
        print(sorted_data.head())
        print(f"Statistics for {player_name_input}:")
        print(player_statistics)
    except ValueError as e:
        print(e)


