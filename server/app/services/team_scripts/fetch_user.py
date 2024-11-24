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
        """
        Initializes an Elasticsearch index if it doesn't exist.
        """
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
        """
        Fetches teams from the Fantasy Premier League for the given league ID.
        """
        base_url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
        teams = []
        page = 26533

        while True:
            url = f"{base_url}?page_standings={page}"
            print(f"Fetching page {page}: {url}")  # Log the current page being fetched

            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for HTTP errors

                data = response.json()
                results = data['standings']['results']
                if not results:  # Break the loop if there are no more results
                    print("No more results. Ending fetch.")
                    break

                for team in results:
                    teams.append({
                        "team_name": team.get('entry_name', 'Unknown Team'),
                        "manager_name": team.get('player_name', 'Unknown Manager'),
                        "team_id": str(team.get('entry', '0')),  # Convert team_id to string
                    })

                print(f"Page {page}: Retrieved {len(results)} teams.")  # Log number of teams retrieved
                page += 1  # Increment the page number to fetch the next page

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data on page {page}: {e}")
                break

            except KeyError as e:
                print(f"Unexpected response structure: Missing key {e}")
                break

        print(f"Total teams fetched: {len(teams)}")
        return teams


    def store_teams_in_elasticsearch(self, index_name, league_id):
        """
        Fetches league teams and stores them in Elasticsearch, including the league ID for tracking.

        Args:
            index_name (str): The Elasticsearch index name.
            league_id (int): The FPL league ID.
        """
        # Check if there are any teams from the league already stored
        '''
        search_body = {
            "query": {
                "match": {
                    "league_id": str(league_id)  # Check for this league_id
                }
            }
        }
        
        response = self.es.search(index=index_name, body=search_body, size=1)
        if response["hits"]["total"]["value"] > 0:
            print(f"League ID {league_id} is already stored in Elasticsearch. Skipping fetch and indexing.")
            return
        '''

        # Fetch and store teams if not already present
        teams = self.fetch_fpl_league_teams(league_id)
        print(len(teams))
        actions = [
            {
                "_op_type": "update",
                "_index": index_name,
                "_id": team["team_id"],
                "doc": {
                    "league_id": str(league_id),  # Add league_id for tracking
                    "team_name": team["team_name"],
                    "manager_name": team["manager_name"],
                    "team_id": team["team_id"],
                },
                "doc_as_upsert": True
            }
            for team in teams
        ]
        bulk(self.es, actions)
        print(f"Data for league ID {league_id} has been stored or updated in Elasticsearch.")


    def search_team_by_name(self, index_name, team_name):
        '''
        Searches Elasticsearch for teams matching the provided team name.
       
        search_body = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {"match_phrase": {"team_name": team_name}},  # Exact match prioritized
                                {"match": {"team_name": {"query": team_name, "fuzziness": "AUTO"}}}  # Fuzzy match allowed
                            ]
                        }
                    },
                    "boost_mode": "multiply",
                    "functions": [
                        {
                            "filter": {"match_phrase": {"team_name": team_name}},
                            "weight": 2
                        },
                        {
                            "filter": {"match": {"team_name": {"query": team_name, "fuzziness": "AUTO"}}},
                            "weight": 1.5
                        }
                    ]
                }
            },
            "size": 10  # Limit results
        }       
        '''
        search_body = {
            "query": {
                "match_phrase": {
                    "team_id": "979930"  # Phrase match for exact text
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
    base_url = "https://fantasy.premierleague.com/api/entry/"
    
    # Fetch team details
    try:
        team_response = requests.get(f"{base_url}{team_id}/")
        team_response.raise_for_status()
        team_info = team_response.json()
        manager_name = team_info.get("player_first_name", "") + " " + team_info.get("player_last_name", "")
        team_name = team_info.get("name", "")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team info: {e}")
        return None

    # Fetch players for the specified gameweek
    try:
        picks_response = requests.get(f"{base_url}{team_id}/event/{gameweek}/picks/")
        picks_response.raise_for_status()
        picks_info = picks_response.json()
        selected_players = [
            {
                "player_id": pick["element"],
                "position": pick["position"],
                "multiplier": pick["multiplier"],
                "is_captain": pick["is_captain"],
                "is_vice_captain": pick["is_vice_captain"],
            }
            for pick in picks_info.get("picks", [])
        ]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team picks: {e}")
        return None

    return {
        "manager_name": manager_name,
        "team_name": team_name,
        "selected_players": selected_players,
    }


def save_team_to_csv(team_data, season="2024-25", gw_directory="GW_1"):
    """
    Saves the Fantasy Premier League team data to a CSV file.

    Args:
        team_data (dict): The data object containing the manager's name, team name, and selected players.
        season (str): The season directory (default: "2024-25").
        gw_directory (str): The game week directory.
    """
    try:
        response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
        response.raise_for_status()
        data = response.json()
        players_info = {player["id"]: player for player in data["elements"]}
        team_names = {team["id"]: team["name"] for team in data["teams"]}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player or team data: {e}")
        return

    csv_data = []
    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    for player in team_data["selected_players"]:
        player_info = players_info.get(player["player_id"])
        if not player_info:
            print(f"Player with ID {player['player_id']} not found.")
            continue

        csv_data.append({
            "full_name": f"{player_info['first_name']} {player_info['second_name']}",
            "position": position_map.get(player_info["element_type"], "Unknown"),
            "team": team_names.get(player_info["team"], "Unknown"),
            "is_captain": 1 if player["is_captain"] else 0,
            "price": player_info["now_cost"] / 10,
            "week1": None,
            "week2": None,
            "week3": None,
            "is_benched": player["multiplier"] == 0,
        })

    file_name = f"{team_data['team_name'].replace(' ', '_')}.csv"
    with open(file_name, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["full_name", "position", "team", "is_captain", "price", "week1", "week2", "week3", "is_benched"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Team data saved successfully as {file_name}")


def main():
    search_instance = Search()

    # User inputs for dynamic configuration
    LEAGUE_ID = 314
    INDEX_NAME = f"fpl_league_{LEAGUE_ID}"
    CURRENT_GW = 12
   
    search_instance.initialize_index(INDEX_NAME)
    search_instance.store_teams_in_elasticsearch(INDEX_NAME, LEAGUE_ID)

    # Search for a team
    team_name = input("Enter the team name to search: ")
    search_instance.search_team_by_name(INDEX_NAME, team_name)

    # Fetch team by ID
    team_id = input("Enter the team ID to fetch details: ")
    fantasy_team = get_fpl_team_by_id(team_id, CURRENT_GW)
    if fantasy_team:
        save_team_to_csv(fantasy_team, f"GW_{CURRENT_GW}")
    else:
        print("No data found for the given team ID.")


if __name__ == "__main__":
    main()


'''
THE COMMAND TO RUN ELASTIC SEARCH IS AS FOLLOWS

../../../elasticsearch/bin/elasticsearch -d 

'''