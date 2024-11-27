import requests

def get_total_players():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        total_players = data.get('total_players')
        return total_players
    else:
        raise Exception('Failed to fetch total number of players.')