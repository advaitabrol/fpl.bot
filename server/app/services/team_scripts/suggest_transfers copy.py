import os
import pandas as pd
from itertools import combinations
from collections import Counter

from app.services.current_gw_service import FPLGWService

# Directory containing the player data
def get_base_dir():
    """
    Dynamically constructs the base directory path using the current gameweek from FPLGWService.
    """
    try:
        # Retrieve the current gameweek
        current_gw = FPLGWService.get_current_gameweek()
        if not current_gw:
            raise ValueError("Unable to determine the current gameweek.")

        # Construct the base directory path
        base_dir = os.path.join(
            os.getcwd(),  # Start from the current working directory
            "prediction_data",  # Root directory for predictions
            "2024-25",  # Example season
            f"GW{current_gw}"  # Current gameweek
        )
        return base_dir
    except Exception as e:
        raise RuntimeError(f"Error retrieving base directory: {str(e)}")

def load_player_data(base_dir):
    """Load player data for all potential transfers, applying minimum expected points thresholds by position."""
    import os
    import pandas as pd

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
        df['name'] = df['name'].str.strip().str.lower()  # Normalize player names

        # Combine week1, week2, and week3 into a single 'expected_points' column
        df['expected_points'] = df[['week1', 'week2', 'week3']].values.tolist()

        # Apply the exclusion criteria based on total expected points
        df['total_expected_points'] = df['expected_points'].apply(sum)  # Sum the points for filtering
        if pos == 'GK':
            df = df[df['total_expected_points'] >= 7]
        elif pos == 'DEF':
            df = df[df['total_expected_points'] >= 9]
        elif pos == 'MID':
            df = df[df['total_expected_points'] >= 13]
        elif pos == 'ATT':
            df = df[df['total_expected_points'] >= 15]

        # Drop the now redundant total_expected_points column
        df = df.drop(columns=['total_expected_points'])

        all_players = pd.concat([all_players, df], ignore_index=True)

    # Ensure required columns are present
    required_columns = {'price', 'name', 'expected_points'}
    if not required_columns.issubset(all_players.columns):
        raise ValueError(f"Required columns {required_columns} not found in player data files.")
    
    return all_players

def load_input_team_from_json(input_team_json):
    """Load input team data from the provided JSON structure."""
    input_team = pd.DataFrame(input_team_json)
    input_team['name'] = input_team['name'].str.strip().str.lower()
    input_team['total_expected_points'] = input_team[['week1', 'week2', 'week3']].sum(axis=1)
    return input_team

def get_position_limits():
    """Defines the constraints for the starting 11 lineup."""
    return {'GK': (1, 1), 'DEF': (3, 5), 'MID': (2, 5), 'ATT': (1, 3)}


def select_optimal_starting_11(team, week, dp_cache):
    """Selects the optimal starting 11 lineup for a given week based on position constraints."""
    # Use both team and week for a unique cache key
    week_key = f"{hash(tuple(player['name'] for player in team))}_week{week}"
    
    if week_key in dp_cache:
        return dp_cache[week_key]

    # Positional constraints (lower and upper limits)
    position_limits = {
        "GK": (1, 1),  # Exactly 1 GK
        "DEF": (3, 5), # Minimum 3, Maximum 5 DEF
        "MID": (2, 5), # Minimum 2, Maximum 5 MID
        "FWD": (1, 3), # Minimum 1, Maximum 3 FWD
    }

    # Separate players by position
    position_candidates = {
        position: sorted(
            [p for p in team if p["position"] == position],
            key=lambda x: x["expected_points"][week - 1],
            reverse=True,
        )
        for position in position_limits
    }
    # Step 1: Select the minimum required players for each position
    starting_11 = []
    for position, (min_limit, _) in position_limits.items():
        starting_11.extend(position_candidates[position][:min_limit])

    # Step 2: Fill remaining slots while respecting maximum positional constraints
    remaining_slots = 11 - len(starting_11)
    all_candidates = sorted(
        [p for p in team if p not in starting_11],
        key=lambda x: x["expected_points"][week - 1],
        reverse=True,
    )

    for candidate in all_candidates:
        if remaining_slots == 0:
            break

        position = candidate["position"]
        current_count = sum(1 for p in starting_11 if p["position"] == position)
        _, max_limit = position_limits[position]

        # Add the candidate if the positional constraint allows it
        if current_count < max_limit:
            starting_11.append(candidate)
            remaining_slots -= 1

    # Sanity check: Ensure exactly 11 players are selected
    assert len(starting_11) == 11, f"Starting 11 does not have exactly 11 players! Found: {len(starting_11)}"

    # Cache the result
    dp_cache[week_key] = starting_11

    return starting_11




def calculate_team_points_with_roles(team, dp_cache, captain_scale=2.0):
    """Calculates the total expected points for the team over 3 weeks, with captain points doubled."""
    try:
        total_points = 0
        for week in range(1, 4):
            starting_11 = select_optimal_starting_11(team, week, dp_cache)
            
            # Find the player with the most expected points to designate as the captain
            captain = max(starting_11, key=lambda player: player['expected_points'][week - 1])
            
            # Calculate the total points for the starting 11
            week_points = sum(player['expected_points'][week - 1] for player in starting_11)
            
            # Double the captain's points
            week_points += captain['expected_points'][week - 1] * (captain_scale - 1)
            
            total_points += week_points

        return total_points
    except Exception as e:
        print("Error in calculate_team_points_with_roles:", str(e))
        raise


def suggest_transfers(input_team_json, max_transfers=2, keep=[], blacklist=[], captain_scale=2.0):
    """
    Suggests optimal transfers for a team to maximize expected points.
    """
    try:
        # Dynamically get the base directory
        base_dir = get_base_dir()

        # Load all potential transfer players
        all_players = load_player_data(base_dir)

        # Convert input team JSON into a DataFrame
        input_team = pd.DataFrame(input_team_json)

        # Adjust captain's expected points
        for player in input_team_json:
            if any(player["isCaptain"]):  # Check if the player is captain in any week
                adjusted_points = [
                    point * (1 - 0.5) if is_captain else point
                    for point, is_captain in zip(player["expected_points"], player["isCaptain"])
                ]
                player["expected_points"] = adjusted_points
        input_team = pd.DataFrame(input_team_json)

        # Exclude current team players from transfer candidates
        current_team_names = set(input_team['name'].str.lower().str.strip())
        all_players = all_players[~all_players['name'].str.lower().str.strip().isin(current_team_names)]

        # Normalize keep and blacklist sets
        keep_set = set(name.strip().lower() for name in keep)
        blacklist_set = set(name.strip().lower() for name in blacklist)

        dp_cache = {}

        # Initialize variables
        current_team = input_team.to_dict('records')

        best_team = current_team.copy()
        best_score = calculate_team_points_with_roles(best_team, dp_cache, captain_scale)

        transfers_suggestion = []

        for t in range(1, max_transfers + 1):
            for out_players in combinations(current_team, t):

                # Get positions and budget of players to transfer out
                out_positions = [player['position'] for player in out_players]
                out_points = sum(sum(player['expected_points']) for player in out_players)
                out_budget = sum(player['price'] for player in out_players)

                # Identify potential candidates for each outgoing position
                position_candidates = {}
                for position in out_positions:
                    position_candidates[position] = all_players[
                        (all_players['position'] == position) &
                        (~all_players['name'].str.lower().str.strip().isin(current_team_names)) &
                        (~all_players['name'].str.lower().str.strip().isin(blacklist_set))
                    ].to_dict('records')

                # Generate combinations of candidates matching the outgoing positions
                for in_players in combinations(sum(position_candidates.values(), []), t):
                    if sorted(player['position'] for player in in_players) != sorted(out_positions):
                        continue

                    # Form the new team
                    new_team = [player for player in current_team if player not in out_players] + list(in_players)
                    total_price = sum(player['price'] for player in new_team)
                    team_counts = Counter(player['team'] for player in new_team)

                    # Validate team constraints
                    if total_price <= 100 and all(count <= 3 for count in team_counts.values()):
                        new_score = calculate_team_points_with_roles(new_team, dp_cache, captain_scale)
                        if new_score > best_score:
                            best_score = new_score
                            best_team = new_team
                            transfers_suggestion = [
                                {"out": out, "in": inp}
                                for out, inp in zip(out_players, in_players)
                            ]
        # Update team JSON with the new structure
        updated_team_json = update_team_json(best_team, dp_cache, captain_scale)

        return updated_team_json, transfers_suggestion

    except Exception as e:
        print("Error in suggest_transfers:", str(e))
        raise


def update_team_json(best_team, dp_cache, captain_scale):
    """Updates the team JSON with expected points, captain, and bench attributes."""
    updated_team = []

    for week in range(1, 4):
        # Get the starting 11 for the current week
        starting_11 = select_optimal_starting_11(best_team, week, dp_cache)
        # Identify the captain for the week
        captain = max(starting_11, key=lambda player: player['expected_points'][week - 1])
        
        for player in best_team:
            # Copy player data to avoid modifying the original
            updated_player = player.copy()
            
            # Initialize isCaptain and isBench arrays if not present
            updated_player.setdefault('isCaptain', [False, False, False])
            updated_player.setdefault('isBench', [False, False, False])
            
            in_starting_11 = updated_player['name'] in {player['name'] for player in starting_11}

            # Update captain and bench status for the current week
            updated_player['isCaptain'][week - 1] = in_starting_11 and updated_player['name'] == captain['name']
            updated_player['isBench'][week - 1] =  not in_starting_11
            
            # Rescale captain's points for the current week
            if updated_player['isCaptain'][week - 1]:
                updated_player['expected_points'][week - 1] *= captain_scale
            
            # Capitalize first and last names after checking membership
            updated_player['name'] = ' '.join(
                word.capitalize() for word in updated_player['name'].split()
            )
            
            # Ensure `totalExpectedPoints` reflects cumulative expected points
            updated_player['totalExpectedPoints'] = sum(updated_player['expected_points'])

            # Avoid duplicate entries in the updated team
            if not any(p['name'] == updated_player['name'] for p in updated_team):
                updated_team.append(updated_player)

    return updated_team









def main():
    # Define the test input JSON
    test_input_json = [{'name': 'David Raya Martin', 'team': 'Arsenal', 'position': 'GK', 'price': 5.6, 'expected_points': [3.47, 3.39, 3.42], 'isBench': [False, False, False], 'isCaptain': [False, False, False]}, {'name': 'Joško Gvardiol', 'team': 'Man City', 'position': 'DEF', 'price': 6.3, 'expected_points': [3.3, 3.34, 3.34], 'isBench': [False, False, False], 'isCaptain': [False, False, False]}, {'name': 'Noussair Mazraoui', 'team': 'Man Utd', 'position': 'DEF', 'price': 4.6, 'expected_points': [3.11, 3.11, 3.1], 'isBench': [False, False, False], 'isCaptain': [False, False, False]}, {'name': 'Gabriel dos Santos Magalhães', 'team': 'Arsenal', 'position': 'DEF', 'price': 6.1, 'expected_points': [3.18, 2.92, 3.01], 'isBench': [False, False, False], 'isCaptain': [False, False, False]}, {'name': 'Morgan Rogers', 'team': 'Aston Villa', 'position': 'MID', 'price': 5.4, 'expected_points': [4.33, 3.59, 4.07], 'isBench': [False, False, False], 'isCaptain': [False, False, False]}, {'name': 'Bryan Mbeumo', 'team': 'Brentford', 'position': 'MID', 'price': 7.9, 'expected_points': [4.4, 4.07, 3.92], 'isBench': [False, False, False], 'isCaptain': [False, False, False]}, {'name': 'James Maddison', 'team': 'Spurs', 'position': 'MID', 'price': 7.6, 'expected_points': [4.23, 4.29, 4.42], 'isBench': [False, False, False], 'isCaptain': [False, False, False]}, {'name': 'Bukayo Saka', 'team': 'Arsenal', 'position': 'MID', 'price': 10.1, 'expected_points': [5.39, 5.07, 5.3], 'isBench': [False, False, False], 'isCaptain': [False, False, False]}, {'name': 'Erling Haaland', 'team': 'Man City', 'position': 'FWD', 'price': 15.2, 'expected_points': [6.83, 6.99, 7.01], 'isBench': [False, False, False], 'isCaptain': [False, False, False]}, {'name': 'Yoane Wissa', 'team': 'Brentford', 'position': 'FWD', 'price': 6.1, 'expected_points': [4.31, 4.1, 3.94], 'isBench': [False, False, False], 'isCaptain': [False, False, False]}, {'name': 'Matheus Santos Carneiro Da Cunha', 'team': 'Wolves', 'position': 'FWD', 'price': 6.8, 'expected_points': [7.3, 6.7, 6.86], 'isBench': [False, False, False], 'isCaptain': [True, True, True]}, {'name': 'Łukasz Fabiański', 'team': 'West Ham', 'position': 'GK', 'price': 4.0, 'expected_points': [2.97, 3.04, 2.99], 'isBench': [True, True, True], 'isCaptain': [False, False, False]}, {'name': 'Brennan Johnson', 'team': 'Spurs', 'position': 'MID', 'price': 6.8, 'expected_points': [4.53, 4.59, 4.62], 'isBench': [True, True, True], 'isCaptain': [False, False, False]}, {'name': 'Ola Aina', 'team': "Nott'm Forest", 'position': 'DEF', 'price': 4.8, 'expected_points': [3.07, 3.06, 2.41], 'isBench': [True, True, True], 'isCaptain': [False, False, False]}, {'name': 'Jacob Greaves', 'team': 'Ipswich', 'position': 'DEF', 'price': 4.0, 'expected_points': [0.0, 0.0, 0.0], 'isBench': [True, True, True], 'isCaptain': [False, False, False]}]

    # Arguments for suggest_transfers
    max_transfers = 2
    keep = []  # No players to keep explicitly
    blacklist = []  # No players blacklisted
    captain_scale = 2.0  # Double the captain's points

    try:
        # Call suggest_transfers
        updated_team_json, transfers_suggestion= suggest_transfers(
            test_input_json, max_transfers, keep, blacklist, captain_scale
        )

        # Output results
        print("\n=== Suggest Transfers Results ===")
        print("Updated Team JSON:")
        for player in updated_team_json:
            print(player)

        print("\nTransfers Suggestion:")
        for out_name, in_name in transfers_suggestion:
            print(f"Transfer Out: {out_name} -> Transfer In: {in_name}")

    except Exception as e:
        print("Error while testing suggest_transfers:", str(e))


if __name__ == "__main__":
    main()
