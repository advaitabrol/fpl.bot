import os
import csv
import requests
from fastapi import HTTPException
from fuzzywuzzy import process
from app.services.current_gw_service import FPLGWService



class TeamService:
    BASE_FPL_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
    PREDICTION_DATA_DIR = os.path.join(os.getcwd(), "prediction_data", "2024-25")

    @staticmethod
    def get_team_details(team_id):
        """
        Fetches and transforms team details into the required format.
        """
        # Fetch raw team data
        team_data = TeamService.fetch_raw_team_data(team_id)

        # Load current prices
        current_prices = TeamService.load_current_prices()

        # Get the current gameweek
        current_gw = FPLGWService.get_current_gameweek()
        if not current_gw:
            raise HTTPException(status_code=500, detail="Unable to fetch current gameweek.")

        # Transform the raw team data
        transformed_team = {
            "team_name": team_data["team_name"],
            "bank": team_data["bank"],
            "team": []
        }

        for player in team_data["selected_players"]:
            player_id = player["player_id"]

            # Fetch player details from the FPL API
            player_details = TeamService.fetch_player_details(player_id)

            # Get player price (using fuzzy matching if necessary)
            player_price = TeamService.get_player_price(player_details["name"], current_prices)

            # Fetch expected points
            expected_points = TeamService.get_expected_points(player_details["name"], player_details["position"], current_gw)
            

            # Add transformed player data
            transformed_team["team"].append({
                "name": player_details["name"],
                "team": player_details["team"],
                "position": player_details["position"],
                "price": player_price,
                "expected_points": [point * 2 if player["is_captain"] else point for point in expected_points],
                "isBench": [player["multiplier"] == 0] * 3,
                "isCaptain": [player["is_captain"]] * 3,
            })

        return transformed_team

    @staticmethod
    def fetch_raw_team_data(team_id):
        """
        Fetches raw team data from the FPL API.
        """
        from app.services.team_scripts.fpl_id import FPLService  # Avoid circular imports
        return FPLService.get_team_by_id(team_id)

    @staticmethod
    def fetch_player_details(player_id):
        """
        Fetches player details (name, team, position) from the FPL API.
        """
        try:
            response = requests.get(TeamService.BASE_FPL_URL)
            response.raise_for_status()
            data = response.json()

            for player in data["elements"]:
                if player["id"] == player_id:
                    return {
                        "name": f"{player['first_name']} {player['second_name']}",
                        "team": TeamService.map_team_id_to_name(player["team"], data),
                        "position": TeamService.map_position(player["element_type"]),
                    }

            raise HTTPException(status_code=404, detail=f"Player with ID {player_id} not found.")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error fetching player details: {e}")

    @staticmethod
    def map_team_id_to_name(team_id, data):
        """
        Maps team ID to team name using FPL data.
        """
        for team in data["teams"]:
            if team["id"] == team_id:
                return team["name"]
        return "Unknown Team"

    @staticmethod
    def map_position(element_type):
        """
        Maps the FPL element type to the desired position format.
        """
        position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        return position_map.get(element_type, "Unknown")

    @staticmethod
    def load_current_prices():
        """
        Loads player prices from `current_prices.csv` into a dictionary.
        """
        prices = {}
        csv_path = os.path.join(os.getcwd(), "current_prices.csv")

        with open(csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row["name"]
                prices[player_name] = float(row["now_cost"]) 
        return prices

    @staticmethod
    def get_player_price(player_name, current_prices):
        """
        Gets the price of a player using fuzzy matching if an exact match is not found.
        """
        # Attempt exact match first
        if player_name in current_prices:
            return current_prices[player_name]

        # Use fuzzy matching to find the closest match
        closest_match, score = process.extractOne(player_name, current_prices.keys(), score_cutoff=80)

        if closest_match:
            return current_prices[closest_match]

        # Return None if no match is found
        return None

    @staticmethod
    def get_expected_points(player_name, position, current_gw):
        """
        Fetches expected points for a player from the `prediction_data` directory by matching the name using fuzzy matching.
        """
        position_file = os.path.join(TeamService.PREDICTION_DATA_DIR, f"GW{current_gw}", f"{position}.csv")
        
        # Check if the position file exists
        if not os.path.exists(position_file):
            return [0.0, 0.0, 0.0]  # Default if file doesn't exist

        # Read the CSV file
        with open(position_file, mode="r", encoding="utf-8") as file:
            reader = list(csv.DictReader(file))  # Convert to a list for fuzzy matching

            # Extract player names for matching
            player_names = [row["name"] for row in reader]

            # Find the closest match using fuzzy matching
            closest_match, score = process.extractOne(player_name, player_names, score_cutoff=80)
            if closest_match:
                # Find the row corresponding to the closest match
                for row in reader:
                    if row["name"] == closest_match:
                        return [
                            float(row.get("week1", 0.0)),
                            float(row.get("week2", 0.0)),
                            float(row.get("week3", 0.0)),
                        ]

        # Default if no close match is found
        return [0.0, 0.0, 0.0]
