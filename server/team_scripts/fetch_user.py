import os
import requests
import csv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pprint import pprint
from fuzzywuzzy import fuzz, process

# Define the Search class
class Search:
    def __init__(self):
        es_password = os.getenv("ELASTIC_PASSWORD")  # Ensure this is set correctly
        self.es = Elasticsearch(
            'https://localhost:9200',  # Use HTTPS
            basic_auth=('elastic', es_password),  # Basic authentication with username and password
            verify_certs=False  # Set to False for testing with self-signed certificates (change for production)
        )
        if self.es.ping():
            print("Connected to Elasticsearch!")
        else:
            print("Could not connect to Elasticsearch.")

    def initialize_index(self, index_name):
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body={
                "mappings": {
                    "properties": {
                        "team_name": {"type": "text"},
                        "manager_name": {"type": "text"},
                        "team_id": {"type": "keyword"}
                    }
                }
            })
            print(f"Index '{index_name}' is ready.")
        else:
            print(f"Index '{index_name}' already exists.")

    
    def fetch_fpl_league_teams(self, league_id):
        base_url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
        teams = []
        page = 1

        while True:
            response = requests.get(f"{base_url}?page_standings={page}")

            if response.status_code == 200:
                data = response.json()
                results = data['standings']['results']
                if not results:  # Break the loop if there are no more results
                    break

                for team in results:
                    teams.append({
                        "team_name": team['entry_name'],
                        "manager_name": team['player_name'],
                        "team_id": str(team['entry']),  # Convert team_id to string
                    })

                page += 1  # Increment the page number to fetch the next page
            else:
                print(f"Error fetching data: {response.status_code}")
                break

        return teams


    def store_teams_in_elasticsearch(self, index_name):
        teams = self.fetch_fpl_league_teams(213)
        actions = [
            {
                "_op_type": "update",
                "_index": index_name,
                "_id": team["team_id"],  # Use team_id as the document ID for upsert
                "doc": {
                    "team_name": team["team_name"],
                    "manager_name": team["manager_name"],
                     "team_id": team["team_id"],
                },
                "doc_as_upsert": True  # Insert if not present, otherwise update
            }
            for team in teams
        ]
        # Use bulk indexing with upsert for efficiency
        bulk(self.es, actions)
        print("Data stored or updated in Elasticsearch.")

    def search_team_by_name(self, index_name, team_name):
        search_body = {
            "query": {
                "match": {
                    "team_name": team_name
                }
            }
        }
        response = self.es.search(index=index_name, body=search_body)
        hits = response['hits']['hits']
        if hits:
            print("Found teams:")
            for hit in hits:
                source = hit['_source']
                print(f"Retrieved document: {source}")  # Debugging line to print the entire document
                # Safely access the team_id field
                team_id = source.get('team_id', 'N/A')  # Use 'N/A' if 'team_id' does not exist
                print(f"Team: {source['team_name']}, Manager: {source['manager_name']}, ID: {team_id}")
        else:
            print("No teams found with that name.")

def get_fpl_team_by_id(team_id, gameweek=1):
    """
    Fetches Fantasy Premier League team data by team ID, including manager's name, team name, and selected players.

    Args:
        team_id (int): The ID of the FPL team.
        gameweek (int): The gameweek to get the player's selected players for (default is 1).

    Returns:
        dict: A dictionary with the manager's name, team's name, and selected players, or None if not found.
    """
    # Base URL for FPL API
    base_url = "https://fantasy.premierleague.com/api/entry/"
    
    # Fetch the team details (manager's name and team name)
    try:
        team_response = requests.get(f"{base_url}{team_id}/")
        team_response.raise_for_status()
        team_info = team_response.json()
        
        # Extract manager's name and team name
        manager_name = team_info.get("player_first_name", "") + " " + team_info.get("player_last_name", "")
        team_name = team_info.get("name", "")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team info: {e}")
        return None
    
    # Fetch the players selected for the given gameweek
    try:
        picks_response = requests.get(f"{base_url}{team_id}/event/{gameweek}/picks/")
        picks_response.raise_for_status()
        picks_info = picks_response.json()
        
        # Extract player selections
        selected_players = []
        for pick in picks_info.get("picks", []):
            selected_players.append({
                "player_id": pick["element"],
                "position": pick["position"],
                "multiplier": pick["multiplier"],
                "is_captain": pick["is_captain"],
                "is_vice_captain": pick["is_vice_captain"]
            })
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team picks: {e}")
        return None
    
    # Create the team data object
    team_data = {
        "manager_name": manager_name,
        "team_name": team_name,
        "selected_players": selected_players
    }
    
    return team_data

def save_team_to_csv(team_data, gw_directory="GW_1"):
    """
    Saves the Fantasy Premier League team data to a CSV file. Each row represents a player with
    the following columns: full name, position, team name, is_captain (binary), price, week1, week2, week3, is_benched.

    Args:
        team_data (dict): The data object containing the manager's name, team name, and selected players.
        gw_directory (str): The game week directory to retrieve week1, week2, and week3 values from.
    """
    # Load all players' and teams' information from the FPL API
    try:
        response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        response.raise_for_status()
        data = response.json()
        players_data = data["elements"]
        teams_data = data["teams"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching players or teams data: {e}")
        return

    # Map player_id to player details and team_id to team name
    players_info = {player["id"]: player for player in players_data}
    team_names = {team["id"]: team["name"] for team in teams_data}
    
    # Prepare data for CSV
    csv_data = []
    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}  # Mapping position ID to abbreviation
    season_folder = "2024-25"  # Hardcoded season folder (changeable if needed)

    for player in team_data["selected_players"]:
        player_id = player["player_id"]
        
        # Look up the player data
        player_info = players_info.get(player_id)
        if not player_info:
            print(f"Player with ID {player_id} not found in FPL data.")
            continue
        
        # Extract relevant information
        player_full_name = f"{player_info['first_name']} {player_info['second_name']}"
        player_position = position_map.get(player_info["element_type"], "Unknown")
        player_team_name = team_names.get(player_info["team"], "Unknown")
        is_captain = 1 if player["is_captain"] else 0
        player_price = player_info["now_cost"] / 10  # Price in actual value
        
        # Determine if the player is benched
        is_benched = player["multiplier"] == 0
        
        # Locate the player's weekly scores (week1, week2, week3) in the specified gameweek directory
        week1, week2, week3 = None, None, None
        position_file_path = os.path.join("prediction_data", season_folder, gw_directory, f"{player_position}.csv")
        
        try:
            with open(position_file_path, mode="r", encoding="utf-8") as pos_file:
                reader = csv.DictReader(pos_file)
                player_names = [row["name"] for row in reader]
                
                # Fuzzy match the player's full name with names in the CSV
                closest_match, match_score = process.extractOne(player_full_name, player_names, scorer=fuzz.token_sort_ratio)
                
                # Reset the reader and search for the matching row to get week values
                pos_file.seek(0)
                next(reader)  # Skip header
                for row in reader:
                    if row["name"] == closest_match:
                        week1 = float(row.get("week1", 0))
                        week2 = float(row.get("week2", 0))
                        week3 = float(row.get("week3", 0))
                        break
        except FileNotFoundError:
            print(f"Position file {position_file_path} not found.")
        except KeyError:
            print(f"Week data columns missing in {position_file_path}.")

        # Add row to CSV data
        csv_data.append({
            "full_name": player_full_name,
            "position": player_position,
            "team": player_team_name,
            "is_captain": is_captain,
            "price": player_price,
            "week1": week1,
            "week2": week2,
            "week3": week3,
            "is_benched": is_benched
        })
    
    # Sort data by week1 points in descending order
    csv_data = sorted(csv_data, key=lambda x: x["week1"], reverse=True)
    
    # Save CSV file
    team_name_sanitized = team_data["team_name"].replace(" ", "_").replace("/", "_")  # Sanitize filename
    file_name = f"{team_name_sanitized}.csv"
    with open(file_name, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["full_name", "position", "team", "is_captain", "price", "week1", "week2", "week3", "is_benched"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Team data saved successfully as {file_name}")



def main():
    # Create an instance of the Search class
    search_instance = Search()
    INDEX_NAME = "fpl_teams"
    CURRENT_GW = 10

    # Initialize and load data into Elasticsearch, only needs to be done once
    #search_instance.initialize_index(INDEX_NAME)
    #search_instance.store_teams_in_elasticsearch(INDEX_NAME)  # This can be scheduled periodically for updates

    # User input to search for a team
    user_team_name = input("Enter the team name you are looking for: ")
    search_instance.search_team_by_name(INDEX_NAME, user_team_name)

    what_id = input("What is the id of team that matches yours: ")
    fantasy_team = get_fpl_team_by_id(what_id, 10)
    save_team_to_csv(fantasy_team, 'GW_10'); 


if __name__ == "__main__":
    main()