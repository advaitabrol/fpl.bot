import requests

def find_team_id_in_overall_league(team_name):
    page = 1
    while True:
        url = f"https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page={page}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            break

        league_data = response.json()
        
        # Iterate through the teams on the current page
        for entry in league_data['standings']['results']:
            if entry['entry_name'].lower() == team_name.lower():
                return entry['entry']  # Return the team ID if the name matches

        # Check if there are more pages of standings
        if not league_data['standings']['has_next']:
            break  # Exit the loop if there are no more pages

        page += 1  # Move to the next page

    return None  # Return None if the team was not found

# Example usage
team_name = "megs>goals fc"  # Replace this with the actual team name
team_id = find_team_id_in_overall_league(team_name)

if team_id:
    print(f"Team ID for '{team_name}': {team_id}")
else:
    print(f"Team '{team_name}' not found in the Overall league.")
