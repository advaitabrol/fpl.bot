import os
import pandas as pd
from itertools import combinations
from collections import Counter
from fuzzywuzzy import fuzz, process

from app.services.current_gw_service import FPLGWService
from app.services.active_users import get_total_players

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
            df = df[df['total_expected_points'] >= 8]
        elif pos == 'DEF':
            df = df[df['total_expected_points'] >= 8]
        elif pos == 'MID':
            df = df[df['total_expected_points'] >= 8]
        elif pos == 'FWD':
            df = df[df['total_expected_points'] >= 8]

        # Drop the now redundant total_expected_points column
        df = df.drop(columns=['total_expected_points'])

        all_players = pd.concat([all_players, df], ignore_index=True)

    # Ensure required columns are present
    required_columns = {'price', 'name', 'expected_points', 'selected'}
    if not required_columns.issubset(all_players.columns):
        raise ValueError(f"Required columns {required_columns} not found in player data files.")

    # Mutate the 'selected' column to represent percentages
    total_players = get_total_players()
    all_players['selected'] = (all_players['selected'] / total_players) * 100
    
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


def suggest_transfers(
    input_team_json,
    max_transfers=2,
    keep_players=[],
    avoid_players=[],
    keep_teams=[],
    avoid_teams=[],
    desired_selected=[],
    captain_scale=2.0,
    bank=0.0
):
    try:
        # Normalize captain's points in the input team
        for player in input_team_json:
            if "isCaptain" in player and isinstance(player["isCaptain"], list):
                player["expected_points"] = [
                    point / captain_scale if is_captain else point
                    for point, is_captain in list(zip(player["expected_points"], player["isCaptain"]))
                ]

        # Dynamically get the base directory
        base_dir = get_base_dir()

        # Load all potential transfer players
        all_players = load_player_data(base_dir)

        # Convert input team JSON into a DataFrame
        input_team = pd.DataFrame(input_team_json)


        # Identify players and teams to keep in the final team
        keep_player_matches = {
            match[0]
            for player in keep_players
            for match in process.extractBests(player, input_team['name'], scorer=fuzz.partial_ratio, limit=len(input_team))
            if match[1] > 50
        }

        keep_team_matches = {
            match[0]
            for team in keep_teams
            for match in process.extractBests(team, input_team['team'], scorer=fuzz.partial_ratio, limit=len(input_team))
            if match[1] > 40
        }

        # Ensure these players are always included in the final team
        required_players = input_team[
            input_team['name'].isin(keep_player_matches) | input_team['team'].isin(keep_team_matches)
        ]

        # Filter out players in `required_players` from the possible transfer-out pool
        current_team = input_team[
            ~input_team['name'].isin(required_players['name'])
        ].to_dict('records')

        # Filter all_players based on avoid criteria
        if avoid_players:
            avoid_player_matches = {
                match[0]
                for player in avoid_players
                for match in process.extractBests(player, all_players['name'], scorer=fuzz.partial_ratio, limit=len(all_players))
                if match[1] > 75
            }
            all_players = all_players[~all_players['name'].str.lower().isin([name.lower() for name in avoid_player_matches])]

        if avoid_teams:
            avoid_team_matches = {
                match[0]
                for team in avoid_teams
                for match in process.extractBests(team, all_players['team'], scorer=fuzz.partial_ratio, limit=len(all_players))
                if match[1] > 75
            }
            all_players = all_players[~all_players['team'].str.lower().isin([team.lower() for team in avoid_team_matches])]

        if desired_selected:
            lower_bound, upper_bound = desired_selected
            all_players = all_players[
                (all_players['selected'] >= lower_bound) & (all_players['selected'] <= upper_bound)
            ]

        # Initialize variables for the best team and score
        dp_cache = {}
        best_team = current_team + required_players.to_dict('records')
        best_score = calculate_team_points_with_roles(best_team, dp_cache, captain_scale)
        transfers_suggestion = []
        remaining_bank = bank

        for t in range(1, max_transfers + 1):
            for out_players in combinations(current_team, t):
                out_positions = [player['position'] for player in out_players]
                out_budget = sum(player['price'] for player in out_players)

                # Identify potential candidates for each outgoing position
                position_candidates = {}
                for position in out_positions:
                    position_candidates[position] = all_players[
                        (all_players['position'] == position)
                    ].to_dict('records')

                for in_players in combinations(sum(position_candidates.values(), []), t):
                    # Ensure the positions match
                    if sorted(player['position'] for player in in_players) != sorted(out_positions):
                        continue

                    # Ensure no player being brought in is already on the team
                    current_team_names = {current_player['name'].strip().lower() for current_player in input_team_json}

                    if any(player['name'].strip().lower() in current_team_names for player in list(in_players)):
                        continue

                    # Form the new team including required players
                    new_team = (
                        [player for player in current_team if player not in out_players] 
                        + list(in_players) 
                        + required_players.to_dict('records')  # Add required players to the team
                    )

                    # Calculate the total price and team constraints
                    net_transfer_cost = sum(player['price'] for player in in_players) - out_budget

                    # Ensure the net transfer cost fits within the available budget
                    if net_transfer_cost <= bank:
                        team_counts = Counter(player['team'] for player in new_team)
                        if all(count <= 3 for count in team_counts.values()):
                            new_score = calculate_team_points_with_roles(new_team, dp_cache, captain_scale)
                            if new_score > best_score:
                                best_score = new_score
                                best_team = new_team
                                transfers_suggestion = [
                                    {"out": out, "in_player": inp}
                                    for out, inp in zip(out_players, in_players)
                                ]
                                remaining_bank = bank - net_transfer_cost


        updated_team_json = update_team_json(best_team, dp_cache, captain_scale)
        return updated_team_json, transfers_suggestion, remaining_bank

    except Exception as e:
        print(f"An error occurred: {e}")
        return [], []



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
    
    #def suggest_transfers(input_team_json,max_transfers=2,keep_players=[],avoid_players=[],
    #keep_teams=[], avoid_teams=[], desired_selected=[], captain_scale=2.0,
    #)
    
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
            test_input_json, max_transfers, keep_players=['Bryan Mbuemo'],keep_teams=['Arsenal'], desired_selected=[20,100]
        )

        # Output results
        print("\n=== Suggest Transfers Results ===")
        print("Updated Team JSON:")
        for player in updated_team_json:
            print(player)

        print("\nTransfers Suggestion:")
        for transfer in transfers_suggestion:
            out_name = transfer['out']['name']  # Extract the name of the player being transferred out
            in_name = transfer['in']['name']   # Extract the name of the player being transferred in
            print(f"Transfer Out: {out_name} -> Transfer In: {in_name}")

    except Exception as e:
        print("Error while testing suggest_transfers:", str(e))


if __name__ == "__main__":
    main()
