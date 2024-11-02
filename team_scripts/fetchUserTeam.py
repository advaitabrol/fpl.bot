import os
import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pprint import pprint

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

def main():
    # Create an instance of the Search class
    search_instance = Search()
    INDEX_NAME = "fpl_teams"

    # Initialize and load data into Elasticsearch
    #search_instance.initialize_index(INDEX_NAME)
    #search_instance.store_teams_in_elasticsearch(INDEX_NAME)  # This can be scheduled periodically for updates

    # User input to search for a team
    user_team_name = input("Enter the team name you are looking for: ")
    search_instance.search_team_by_name(INDEX_NAME, user_team_name)

if __name__ == "__main__":
    main()