import os

from fastapi import APIRouter
from app.services.current_gw_service import FPLService
from app.services.wildcard_team import build_wildcard_team
from app.services.freehit_team import build_freehit_team

router = APIRouter()

@router.get("/wildcard-team")
def get_wildcard_team():
    """
    Endpoint to fetch the optimal team for the current game week.
    """
    # Use the FPLService to fetch the current game week
    current_gw = FPLService.get_current_gameweek()
    if not current_gw:
        return {"error": "Unable to determine the current game week from the FPL API."}

    # Dynamically construct the BASE_DIR
    BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # Go up one directory
    "services",  # Navigate to the services folder
    "prediction_data", 
    "2024-25", 
    f"GW{current_gw}")

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
    # Use the FPLService to fetch the current game week
    current_gw = FPLService.get_current_gameweek()
    if not current_gw:
        return {"error": "Unable to determine the current game week from the FPL API."}

    # Dynamically construct the BASE_DIR
    BASE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),  # Go up one directory
        "services",  # Navigate to the services folder
        "prediction_data", 
        "2024-25", 
        f"GW{current_gw}"
    )

    try:
        # Build the freehit team for the current GW
        team_json = build_freehit_team(BASE_DIR)  # Call the renamed function
        return team_json
    except Exception as e:
        return {"error": str(e)}