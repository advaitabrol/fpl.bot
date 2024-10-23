import os
import random
import csv

# Directory to save the generated files
output_dir = './player_data/'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Expanded name lists to generate more unique names
first_names = ['John', 'David', 'Alex', 'Chris', 'Michael', 'James', 'Tom', 'Steve', 'Mark', 'Sam', 
               'Charlie', 'Daniel', 'Matt', 'Adam', 'Ben', 'Jack', 'Josh', 'Luke', 'Nathan', 'Oliver', 
               'Ryan', 'Jacob', 'Noah', 'Ethan', 'Henry', 'Max', 'Leo', 'Oscar', 'Liam']

last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Martinez', 'Wilson', 
              'Moore', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Young', 
              'King', 'Scott', 'Green', 'Baker', 'Adams', 'Clark', 'Evans', 'Collins', 'Turner', 'Bell']

# List of teams (each team can have a maximum of 27 players total)
teams = [
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Chelsea', 'Crystal Palace', 'Everton',
    'Fulham', 'Ipswich Town', 'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
    'Newcastle United', 'Nottingham Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolverhampton'
]

# Dictionary to track the number of players assigned to each team
team_player_count = {team: 0 for team in teams}
max_players_per_team = 27

# Function to generate a random player name
def generate_player_name():
    return f"{random.choice(first_names)} {random.choice(last_names)}"

# Function to generate a random value with a skew towards smaller values
def generate_value():
    return round(random.triangular(4.4, 10, 4.8), 1)  # Skew towards lower values

# Function to generate random expected points with a correlation to value
def generate_expected_points(value):
    if value <= 5.5:
        return round(random.uniform(0, 6), 1)  # Lower value players have lower expected points
    elif value <= 7.0:
        return round(random.uniform(4, 12), 1)  # Mid-value players have mid-range expected points
    else:
        return round(random.uniform(10, 20), 1)  # Higher value players have higher expected points

# Function to assign a player to a team ensuring no team exceeds the 27-player limit
def assign_team():
    available_teams = [team for team, count in team_player_count.items() if count < max_players_per_team]
    if not available_teams:
        raise ValueError("No more teams available with capacity.")
    chosen_team = random.choice(available_teams)
    team_player_count[chosen_team] += 1
    return chosen_team

# Function to generate player data
def generate_players(position, count):
    players = []
    generated_names = set()  # To avoid duplicate names
    while len(players) < count:
        name = generate_player_name()
        # Allow some repeated names if we hit the limit of unique names
        if len(generated_names) < len(first_names) * len(last_names):
            if name not in generated_names:
                generated_names.add(name)
        # Assign team and ensure no team exceeds 27 players
        team_name = assign_team()
        value = generate_value()
        expected_points = generate_expected_points(value)
        players.append([name, team_name, value, expected_points])
    return players

# Function to write players to a CSV file
def write_players_to_csv(position, players):
    output_file = os.path.join(output_dir, f'{position}.csv')
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Player Name', 'Team', 'Value', 'Expected Points'])  # Header row
        writer.writerows(players)
    print(f"Generated {len(players)} players for {position} and saved to {output_file}")

# Define the number of players for each position
positions = {
    'GK': 75,
    'DEF': 165,
    'MID': 165,
    'ATT': 135
}

# Generate and save player data for each position
def main():
    for position, count in positions.items():
        print(f"Generating data for {position}...")
        players = generate_players(position, count)
        print(f"Finished generating {len(players)} players for {position}. Now saving to CSV.")
        write_players_to_csv(position, players)
    print("All files generated successfully!")

if __name__ == "__main__":
    main()
