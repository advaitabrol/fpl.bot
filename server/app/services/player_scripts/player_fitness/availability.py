import requests
import pandas as pd
import schedule
import time

def fetch_fpl_data():
    # URL for the Fantasy Premier League API
    fpl_api_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    # Request data from the FPL API
    response = requests.get(fpl_api_url)
    data = response.json()
    
    # Map team IDs to team names
    team_map = {team['id']: team['name'] for team in data['teams']}
    
    # Extract relevant data for each player
    players_data = []
    for player in data['elements']:
        # Combine first name and second name for full_name
        full_name = f"{player['first_name']} {player['second_name']}"
        
        # Get team name from the team_map
        team_name = team_map.get(player['team'], "Unknown Team")
        
        # Determine the status based on chance of playing values
        if player['chance_of_playing_next_round'] == 100:
            status = "Available"
        elif player['chance_of_playing_next_round'] is None and player['chance_of_playing_this_round'] is None:
            status = "Available"
        elif player['chance_of_playing_next_round'] == 0:
            status = "Unavailable"
        else:
            status = "Doubtful"
        
        # Format chance of playing values, with default to 100% if not specified
        chance_of_playing_next_round = player['chance_of_playing_next_round'] if player['chance_of_playing_next_round'] is not None else 100.0
        chance_of_playing_this_round = player['chance_of_playing_this_round'] if player['chance_of_playing_this_round'] is not None else 100.0
        
        # Append player info to the list
        player_info = {
            "full_name": full_name,
            "team": team_name,
            "status": status,
            "chance_of_playing_next_round": float(chance_of_playing_next_round),
            "chance_of_playing_this_round": float(chance_of_playing_this_round)
        }
        players_data.append(player_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(players_data)
    
    # Save DataFrame to CSV with desired formatting
    df.to_csv("current_availability.csv", index=False)
    print("FPL player data saved to 'current_availability.csv'")



if __name__ == "__main__":

    fetch_fpl_data(); 
   
    '''
    # Schedule the function to run on Monday, Wednesday, and Friday at 8:00 AM
    schedule.every().monday.at("02:00").do(fetch_fpl_data)
    schedule.every().wednesday.at("02:00").do(fetch_fpl_data)
    schedule.every().friday.at("02:00").do(fetch_fpl_data)

    # Run the scheduler
    while True:
        schedule.run_pending()
        time.sleep(100)
    '''