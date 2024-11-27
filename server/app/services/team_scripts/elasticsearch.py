import os
from elasticsearch import Elasticsearch, helpers
from fastapi import HTTPException
import requests


class SearchService:
    def __init__(self):
        es_password = os.getenv("ELASTIC_PASSWORD")  # Load the Elasticsearch password
        self.es = Elasticsearch(
            'https://localhost:9200',
            basic_auth=('elastic', es_password),
            verify_certs=False
        )
        if not self.es.ping():
            raise HTTPException(status_code=500, detail="Could not connect to Elasticsearch")

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
            return f"Index '{index_name}' is ready."
        return f"Index '{index_name}' already exists."

    def fetch_fpl_league_teams(self, league_id):
        """
        Fetch teams from the Fantasy Premier League API for a given league ID.
        """
        base_url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
        teams = []
        page = 1  # Start fetching from the first page

        while True:
            url = f"{base_url}?page_standings={page}"
            try:
                response = requests.get(url)
                response.raise_for_status()

                data = response.json()
                results = data.get("standings", {}).get("results", [])
                if not results:
                    break  # Exit the loop when no more results are found

                for team in results:
                    teams.append({
                        "team_name": team.get('entry_name', 'Unknown Team'),
                        "manager_name": team.get('player_name', 'Unknown Manager'),
                        "team_id": str(team.get('entry', '0')),  # Ensure team_id is a string
                    })

                page += 1  # Fetch the next page
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=500, detail=f"Error fetching data: {e}")
            except KeyError as e:
                raise HTTPException(status_code=500, detail=f"Unexpected response structure: {e}")

        return teams

    def store_teams_in_elasticsearch(self, index_name, league_id):
        """
        Fetch league teams and store them in Elasticsearch.
        """
        teams = self.fetch_fpl_league_teams(league_id)
        actions = [
            {
                "_op_type": "update",
                "_index": index_name,
                "_id": team["team_id"],
                "doc": team,
                "doc_as_upsert": True  # Upsert (update if exists, insert if not)
            }
            for team in teams
        ]
        helpers.bulk(self.es, actions)
        return f"Stored or updated {len(teams)} teams in Elasticsearch."

    def search_team_by_name(self, index_name, team_name):
        """
        Search for teams in Elasticsearch by name.
        """
        search_body = {
            "size": 5,
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {"match_phrase": {"team_name": team_name}},
                                {"match": {"team_name": {"query": team_name, "fuzziness": "AUTO"}}}
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
                            "filter": {"match": {"team_name": {"query": team_name, "fuzziness": 1}}},
                            "weight": 1.5
                        }
                    ]
                }
            }
        }   
        try:
            response = self.es.search(index=index_name, body=search_body)
            return [
                {
                    "team_name": hit["_source"]["team_name"],
                    "manager_name": hit["_source"]["manager_name"],
                    "team_id": hit["_source"]["team_id"]
                }
                for hit in response["hits"]["hits"]
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error searching team: {str(e)}")

    @staticmethod
    def main(league_id):
        """
        Main method to populate Elasticsearch with teams from a given league ID.
        """
        service = SearchService()
        index_name = f"fpl_league_{league_id}"

        print("Initializing Elasticsearch index...")
        init_message = service.initialize_index(index_name)
        print(init_message)

        print(f"Populating Elasticsearch with teams from league ID: {league_id}")
        store_message = service.store_teams_in_elasticsearch(index_name, league_id)
        print(store_message)

if __name__ == "__main__":
    LEAGUE_ID = 314  # Replace with your desired league ID
    SearchService.main(LEAGUE_ID)