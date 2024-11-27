import requests
from fastapi import HTTPException
from app.services.current_gw_service import FPLGWService  # Import the gameweek service


class FPLService:
    @staticmethod
    def get_team_by_id(team_id, gameweek=None):
        """
        Fetch a Fantasy Premier League team by its ID.

        Args:
            team_id (int): The ID of the team.
            gameweek (int, optional): The gameweek to fetch data for. Defaults to the current gameweek.

        Returns:
            dict: A dictionary with the team manager name, team name, and selected players.
        """
        base_url = "https://fantasy.premierleague.com/api/entry/"

        # If no gameweek is provided, fetch the current gameweek
        if gameweek is None:
            gameweek = FPLGWService.get_current_gameweek()
            if not gameweek:
                raise HTTPException(
                    status_code=500,
                    detail="Unable to determine the current gameweek from the FPL API."
                )

        try:
            # Fetch team details
            team_response = requests.get(f"{base_url}{team_id}/")
            team_response.raise_for_status()
            team_info = team_response.json()

            # Fetch players for the specified gameweek
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
            return {
                "manager_name": f"{team_info.get('player_first_name')} {team_info.get('player_last_name')}",
                "team_name": team_info.get("name"),
                "selected_players": selected_players,
            }
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error fetching team info: {e}")
