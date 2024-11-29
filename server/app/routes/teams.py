import os
import json

from typing import List, Tuple, Optional  # Ensure this is imported
from pydantic import BaseModel, ValidationError
from fastapi import APIRouter, HTTPException

from app.services.current_gw_service import FPLGWService
from app.services.team_scripts.wildcard_team import build_wildcard_team
from app.services.team_scripts.freehit_team import build_freehit_team

from app.services.team_scripts.elasticsearch import SearchService
from app.services.team_scripts.fpl_id import FPLService
from app.services.team_scripts.team_json import TeamService
from app.services.team_scripts.optimize_team import optimize_team
from app.services.team_scripts.suggest_transfers import suggest_transfers

router = APIRouter()


team_search = SearchService()
id_search = FPLService()

# Define input model for the optimize-team route
class Player(BaseModel):
    name: str
    team: str
    position: str
    price: float
    expected_points: List[float]
    isBench: List[bool]
    isCaptain: List[bool]

class TransferInput(BaseModel):
    team: List[Player]
    max_transfers: int
    keep_players: List[str] = []
    avoid_players: List[str] = []
    keep_teams: List[str] = []
    avoid_teams: List[str] = []
    desired_selected: Tuple[int, int] = (0, 100)  # A range of selected percentages
    captain_scale: float = 2.0
    bank: float = 0.0


class OptimizeTeamInput(BaseModel):
    team: List[Player]


@router.get("/search-team-name")
def search_team_name(team_name: str, league_id: int = 314):
    """
    Search for teams by name.
    """
    index_name = f"fpl_league_{league_id}"
    try:
        teams = team_search.search_team_by_name(index_name, team_name)
        if not teams:
            return {"message": "No teams found matching the given name."}
        return {"teams": teams}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/your-team")
def your_team(team_id: str):
    """
    Fetch a team by its ID and transform the output using the TeamService.
    """
    try:
        transformed_team = TeamService.get_team_details(team_id)
        return transformed_team
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wildcard-team")
def get_wildcard_team():
    """
    Endpoint to fetch the optimal team for the current game week.
    """
    # Use the FPLGWService to fetch the current game week
    current_gw = FPLGWService.get_current_gameweek()
    if not current_gw:
        return {"error": "Unable to determine the current game week from the FPL API."}

    # Dynamically construct the BASE_DIR
    BASE_DIR = os.path.join(
        os.getcwd(),  # Start from the current working directory
        "prediction_data",  # Directly reference the folder
        "2024-25", 
        f"GW{current_gw}"
    )


    try:
        # Build the optimal team for the current GW
        team_json = build_wildcard_team(BASE_DIR)
        return team_json
    except Exception as e:
        return {"error": str(e)}


@router.get("/freehit-team")
def get_freehit_team():
    """
    Endpoint to fetch the freehit team for the current game week.
    """
    # Use the FPLGWService to fetch the current game week
    current_gw = FPLGWService.get_current_gameweek()
    if not current_gw:
        return {"error": "Unable to determine the current game week from the FPL API."}

    # Dynamically construct the BASE_DIR
    BASE_DIR = os.path.join(
        os.getcwd(),  # Start from the current working directory
        "prediction_data",  # Directly reference the folder
        "2024-25", 
        f"GW{current_gw}"
    )

    try:
        # Build the freehit team for the current GW
        team_json = build_freehit_team(BASE_DIR)  # Call the renamed function
        return team_json
    except Exception as e:
        return {"error": str(e)}

@router.post("/optimize-team")
def optimize_team_route(input_team: OptimizeTeamInput):
    """
    Optimize the given team to maximize expected points.
    """
    try:
        # Convert the input team data into the expected JSON format
        team_json = {"team": [player.dict() for player in input_team.team]}

        # Call the optimize_team function
        optimized_team = optimize_team(team_json)

        return optimized_team
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing team: {str(e)}")

@router.post("/suggest-transfers")
def suggest_transfers_route(input_data: TransferInput):
    """
    Suggest optimal transfers for the given team.
    """
    try:
        # Convert team players to dict format
        team_json = [player.dict() for player in input_data.team]

        # Call the suggest_transfers function
        optimized_team, transfers_suggestion, bank = suggest_transfers(
            input_team_json=team_json,
            max_transfers=input_data.max_transfers,
            keep_players=input_data.keep_players,
            avoid_players=input_data.avoid_players,
            keep_teams=input_data.keep_teams,
            avoid_teams=input_data.avoid_teams,
            desired_selected=input_data.desired_selected,
            captain_scale=input_data.captain_scale,
            bank=input_data.bank,
        )

        # Prepare and return the response
        response = {
            "optimized_team": optimized_team,
            "transfers_suggestion": transfers_suggestion,
            "bank": bank,
        }
        return response

    except ValidationError as ve:
        print("Validation error:", ve.json())
        raise HTTPException(status_code=422, detail=ve.errors())

    except Exception as e:
        print("Error in suggest_transfers_route:", str(e))
        raise HTTPException(status_code=500, detail=f"Error suggesting transfers: {str(e)}")
