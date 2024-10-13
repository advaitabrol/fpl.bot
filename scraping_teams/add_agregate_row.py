import os
import pandas as pd

def add_sum_row_to_csv(df):
    """Sum the numeric columns and add the sum as a new row to the DataFrame."""
    # Columns to sum (all columns except the first two: 'h_a' and 'date' are not numeric)
    numeric_columns = df.columns.difference(['h_a', 'date', 'result'])
    
    # Sum the numeric columns
    summed_row = df[numeric_columns].sum(axis=0)
    
    # Create a new DataFrame for the summed row, with placeholders for non-numeric columns
    summed_row_df = pd.DataFrame([{
        'h_a': 'Total',
        'xG': summed_row['xG'],
        'xGA': summed_row['xGA'],
        'npxG': summed_row['npxG'],
        'npxGA': summed_row['npxGA'],
        'ppda': summed_row['ppda'],
        'ppda_allowed': summed_row['ppda_allowed'],
        'deep': summed_row['deep'],
        'deep_allowed': summed_row['deep_allowed'],
        'scored': summed_row['scored'],
        'missed': summed_row['missed'],
        'xpts': summed_row['xpts'],
        'result': 'Total',  # Placeholder for non-numeric field
        'date': 'N/A',      # Placeholder for non-numeric field
        'wins': summed_row['wins'],
        'draws': summed_row['draws'],
        'loses': summed_row['loses'],
        'pts': summed_row['pts'],
        'npxGD': summed_row['npxGD']
    }])

    # Append the summed row to the original DataFrame
    df_with_sum = pd.concat([df, summed_row_df], ignore_index=True)
    
    return df_with_sum

def process_existing_csv_files(input_folder):
    """Read the existing CSV files from the local folder, add a sum row, and overwrite the file with the new data."""
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            # Construct full file path
            file_path = os.path.join(input_folder, file_name)
            
            # Load the CSV into a pandas DataFrame
            df = pd.read_csv(file_path)

            # Add the sum row to the DataFrame
            df_with_sum = add_sum_row_to_csv(df)

            # Overwrite the existing CSV file with the updated data
            df_with_sum.to_csv(file_path, index=False)

            print(f"Updated {file_name} with summed totals")

def main():
    # Specify the folder where the team CSV files are stored
    input_folder = 'fpl_team_data'  # Directory containing the existing CSV files
    
    # Process all existing CSV files and add the sum row
    process_existing_csv_files(input_folder)

if __name__ == "__main__":
    main()
