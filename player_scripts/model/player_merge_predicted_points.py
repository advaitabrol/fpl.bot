import pandas as pd

def merge_player_weeks(input_csv, output_csv):
    # Read in the CSV file
    df = pd.read_csv(input_csv)
    
    # Sort the dataframe by name to ensure week1, week2, week3 order
    df = df.sort_values(['name']).reset_index(drop=True)
    
    # Create an empty list to store the transformed rows
    merged_rows = []

    # Group by 'name', assuming each player has exactly three rows in the data
    for name, group in df.groupby('name'):
        # Ensure the group is sorted as per appearance in the original CSV for correct week mapping
        group = group.sort_index()
        
        # Extract player information (consistent across rows)
        team = group['team'].iloc[0]
        price = group['price'].iloc[0]
        
        # Extract the predicted points for each "week" column
        week1, week2, week3 = group['predicted_next_week_points'].values
        
        # Create a dictionary for the merged row
        merged_row = {
            'name': name,
            'team': team,
            'price': price,
            'week1': week1,
            'week2': week2,
            'week3': week3
        }
        
        # Append the merged row to the list
        merged_rows.append(merged_row)

    # Convert the list of merged rows to a DataFrame
    merged_df = pd.DataFrame(merged_rows)
    
    # Save the transformed data to a new CSV file
    merged_df.to_csv(output_csv, index=False)
    print(f"Data has been saved to {output_csv}")

# Example usage
merge_player_weeks('prediction_data/2024-25/GW10-12/def.csv', 'prediction_data/2024-25/GW10-12/final_def.csv')
merge_player_weeks('prediction_data/2024-25/GW10-12/gk.csv', 'prediction_data/2024-25/GW10-12/final_gk.csv')
merge_player_weeks('prediction_data/2024-25/GW10-12/mid.csv', 'prediction_data/2024-25/GW10-12/final_mid.csv')
merge_player_weeks('prediction_data/2024-25/GW10-12/fwd.csv', 'prediction_data/2024-25/GW10-12/final_fwd.csv')