import pulp as pl
import pandas as pd

def optimize_team(input_team_json):
    """
    Optimizes the team to maximize expected points while adhering to constraints.

    Parameters:
        input_team_json (dict): Input team in JSON format.

    Returns:
        dict: Optimized team JSON with updated isBench and isCaptain arrays and debug info.
    """

    team_df = pd.DataFrame(input_team_json["team"])
    num_weeks = 3


    # Adjust expected points to remove captain multiplier before optimization
    for w in range(num_weeks):
        captain_mask = team_df["isCaptain"].apply(lambda x: x[w])

        team_df.loc[captain_mask, "expected_points"] = team_df.loc[captain_mask, "expected_points"].apply(
            lambda points: [p / 2 if i == w else p for i, p in enumerate(points)]
        )

    # Initialize LP problem
    prob = pl.LpProblem("OptimizeTeam", pl.LpMaximize)
    num_players = len(team_df)

    # Variables
    start_vars = [[pl.LpVariable(f"start_{w}_{i}", cat='Binary') for i in range(num_players)] for w in range(num_weeks)]
    captain_vars = [[pl.LpVariable(f"captain_{w}_{i}", cat='Binary') for i in range(num_players)] for w in range(num_weeks)]

    # Objective: Maximize total expected points, considering captains
    total_expected_points = pl.lpSum(
        start_vars[w][i] * team_df.iloc[i]["expected_points"][w] +
        captain_vars[w][i] * team_df.iloc[i]["expected_points"][w] * 2  # Apply 2x multiplier for captains during optimization
        for w in range(num_weeks) for i in range(num_players)
    )
    prob += total_expected_points

    # Constraints for each week
    for w in range(num_weeks):
        prob += pl.lpSum(start_vars[w]) == 11  # Exactly 11 players in the starting lineup
        prob += pl.lpSum(start_vars[w][i] for i in range(num_players) if team_df.iloc[i]["position"] == "GK") == 1  # 1 GK
        prob += pl.lpSum(start_vars[w][i] for i in range(num_players) if team_df.iloc[i]["position"] == "DEF") >= 3  # At least 3 DEF
        prob += pl.lpSum(start_vars[w][i] for i in range(num_players) if team_df.iloc[i]["position"] == "MID") >= 2  # At least 2 MID
        prob += pl.lpSum(start_vars[w][i] for i in range(num_players) if team_df.iloc[i]["position"] == "FWD") >= 1  # At least 1 FWD
        prob += pl.lpSum(captain_vars[w]) == 1  # Exactly 1 captain

        for i in range(num_players):
            prob += captain_vars[w][i] <= start_vars[w][i]  # Captain must be in the starting lineup

    # Solve the problem
    prob.solve()

    # Check if a solution was found
    if prob.status != pl.LpStatusOptimal:
        return {"error": "No optimal solution found", "debug_info": debug_info}

    # Update the input JSON with optimized data
    for w in range(num_weeks):
        for i in range(num_players):
            is_bench_value = start_vars[w][i].varValue == 0
            is_captain_value = captain_vars[w][i].varValue == 1
            team_df.at[i, "isBench"][w] = is_bench_value
            team_df.at[i, "isCaptain"][w] = is_captain_value

            # Restore captain multiplier in expected points for output
            if is_captain_value:
                team_df.at[i, "expected_points"][w] *= 2


    input_team_json["team"] = team_df.to_dict(orient="records")
    return {"optimal": input_team_json}
