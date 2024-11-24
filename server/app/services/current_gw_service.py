import requests

class FPLService:
    """
    Service to interact with the Fantasy Premier League API.
    """

    BASE_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

    @staticmethod
    def get_current_gameweek():
        """
        Fetches the current game week from the Fantasy Premier League API.

        Returns:
            int: The current game week ID if successful.
            None: If unable to fetch the current game week.
        """
        try:
            response = requests.get(FPLService.BASE_URL)
            response.raise_for_status()  # Raise an error for HTTP status codes >= 400
            data = response.json()

            # Find the event (game week) where 'is_current' is True
            current_gw = next(
                (event['id'] for event in data['events'] if event.get('is_current')),
                None
            )

            if current_gw:
                return current_gw
            else:
                print("No current game week found.")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from FPL API: {e}")
            return None

