import os
import json

from typing import List  # Ensure this is imported
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
    team: List[Player]  # Reuse the Player model from your existing code
    max_transfers: int = 1
    keep: List[str] = []  # Optional, default to empty list
    blacklist: List[str] = []  # Optional, default to empty list


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
        team_json = [player.dict() for player in input_data.team]

        # Step 3: Extract other parameters
        max_transfers = input_data.max_transfers
        keep = input_data.keep
        blacklist = input_data.blacklist

        # Step 4: Call the suggest_transfers function
        from app.services.team_scripts.suggest_transfers import suggest_transfers
        optimized_team, transfers_suggestion = suggest_transfers(
            input_team_json=team_json,
            max_transfers=max_transfers,
            keep=keep,
            blacklist=blacklist
        )

        # Step 5: Prepare and return the response
        response = {
            "optimized_team": optimized_team,
            "transfers_suggestion": transfers_suggestion,
        }
        return response

    except ValidationError as ve:
        print("Validation error:", ve.json())
        raise HTTPException(status_code=422, detail=ve.errors())

    except Exception as e:
        print("Error in suggest_transfers_route:", str(e))
        raise HTTPException(status_code=500, detail=f"Error suggesting transfers: {str(e)}")
