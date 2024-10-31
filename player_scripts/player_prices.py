import requests
import pandas as pd
import io

import requests
import pandas as pd
import io

def fetch_and_process_csv(url, output_file):
    """
    Fetches a CSV file from the provided URL, processes the 'first_name', 'second_name', and 'now_cost' columns,
    combines 'first_name' and 'second_name' into a single 'name' column, and processes the 'now_cost' column
    by dividing it by 10 and rounding to one decimal place. The processed data is saved into an output CSV file.

    Parameters:
    - url: The URL of the CSV file to fetch.
    - output_file: The name of the output CSV file to save the processed data.
    """

    # Fetch the CSV file from the URL
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch the CSV file. Status code: {response.status_code}")
        return
    
    # Read the CSV data into a DataFrame using io.StringIO
    df = pd.read_csv(io.StringIO(response.text))
    
    # Ensure that the necessary columns are present
    if all(col in df.columns for col in ['first_name', 'second_name', 'now_cost']):
        # Create a new 'name' column by combining 'first_name' and 'second_name'
        df['name'] = df['first_name'] + ' ' + df['second_name']

        # Extract the 'name' and 'now_cost' columns
        df_filtered = df[['name', 'now_cost']]

        # Process the 'now_cost' column (divide by 10 and round to one decimal place)
        df_filtered['now_cost'] = (df_filtered['now_cost'] / 10).round(1)

        # Save the processed data to the output CSV file
        df_filtered.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    else:
        print("The required columns ('first_name', 'second_name', 'now_cost') are not present in the CSV file.")

# Example usage
url = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/cleaned_players.csv'
output_file = 'current_prices.csv'
fetch_and_process_csv(url, output_file)