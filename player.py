#Player Class if there ends up being a use for it

class Player:
    def __init__(self, name, assists, bonus, bps, clean_sheets, creativity, element_code, end_cost, 
                 expected_assists, expected_goal_involvements, expected_goals, expected_goals_conceded, 
                 goals_conceded, goals_scored, ict_index, influence, minutes, own_goals, 
                 penalties_missed, penalties_saved, red_cards, saves, season_name, start_cost, 
                 starts, threat, total_points, yellow_cards):
        self.name = name
        self.assists = assists
        self.bonus = bonus
        self.bps = bps
        self.clean_sheets = clean_sheets
        self.creativity = creativity
        self.element_code = element_code
        self.end_cost = end_cost
        self.expected_assists = expected_assists
        self.expected_goal_involvements = expected_goal_involvements
        self.expected_goals = expected_goals
        self.expected_goals_conceded = expected_goals_conceded
        self.goals_conceded = goals_conceded
        self.goals_scored = goals_scored
        self.ict_index = ict_index
        self.influence = influence
        self.minutes = minutes
        self.own_goals = own_goals
        self.penalties_missed = penalties_missed
        self.penalties_saved = penalties_saved
        self.red_cards = red_cards
        self.saves = saves
        self.season_name = season_name
        self.start_cost = start_cost
        self.starts = starts
        self.threat = threat
        self.total_points = total_points
        self.yellow_cards = yellow_cards

    def __repr__(self):
        return f"Player(name={self.name}, assists={self.assists}, bonus={self.bonus}, bps={self.bps}, clean_sheets={self.clean_sheets}, creativity={self.creativity}, element_code={self.element_code}, end_cost={self.end_cost}, expected_assists={self.expected_assists}, expected_goal_involvements={self.expected_goal_involvements}, expected_goals={self.expected_goals}, expected_goals_conceded={self.expected_goals_conceded}, goals_conceded={self.goals_conceded}, goals_scored={self.goals_scored}, ict_index={self.ict_index}, influence={self.influence}, minutes={self.minutes}, own_goals={self.own_goals}, penalties_missed={self.penalties_missed}, penalties_saved={self.penalties_saved}, red_cards={self.red_cards}, saves={self.saves}, season_name={self.season_name}, start_cost={self.start_cost}, starts={self.starts}, threat={self.threat}, total_points={self.total_points}, yellow_cards={self.yellow_cards})"
