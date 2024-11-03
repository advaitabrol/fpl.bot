import requests
import pandas as pd
import schedule
import time

# Define the function to scrape FPL data and save it as a CSV file
def fetch_fpl_data():
    # URL for the Fantasy Premier League API
    fpl_api_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    # Request data from the FPL API
    response = requests.get(fpl_api_url)
    data = response.json()
    
    # Extract relevant data for each player
    players_data = []
    for player in data['elements']:
        player_info = {
            "First Name": player['first_name'],
            "Second Name": player['second_name'],
            "Availability": "Available" if player['chance_of_playing_next_round'] is None and player['chance_of_playing_this_round'] is None else "Unavailable",
            "Chance of Playing (Next Round)": player['chance_of_playing_next_round'] if player['chance_of_playing_next_round'] is not None else "100%",
            "Chance of Playing (This Round)": player['chance_of_playing_this_round'] if player['chance_of_playing_this_round'] is not None else "100%"
        }
        players_data.append(player_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(players_data)
    
    # Save DataFrame to CSV
    df.to_csv("fpl_player_availability.csv", index=False)
    print("FPL player data saved to 'fpl_player_availability.csv'")

# Schedule the function to run on Monday, Wednesday, and Friday at 8:00 AM
schedule.every().monday.at("02:00").do(fetch_fpl_data)
schedule.every().wednesday.at("02:00").do(fetch_fpl_data)
schedule.every().friday.at("02:00").do(fetch_fpl_data)

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(100)


input_team = [
        "Emiliano Martínez Romero", "Trent Alexander-Arnold", "Diogo Dalot Teixeira", 
        "Micky van de Ven", "Brennan Johnson", "Cole Palmer", "James Maddison", 
        "Ross Barkley", "Nicolas Jackson", "Jhon Durán", "Erling Haaland", 
        "Łukasz Fabiański", "Nathan Wood-Gordon", "Harry Winks", "Sepp van den Berg"
    ]