import os
import pandas as pd
import io
import requests
from concurrent.futures import ThreadPoolExecutor

def fetch_and_process_csv(url, output_file):
    """
    Fetches a CSV file from the provided URL, processes the 'first_name', 'second_name', and 'now_cost' columns,
    combines 'first_name' and 'second_name' into a single 'name' column, and processes the 'now_cost' column
    by dividing it by 10 and rounding to one decimal place. The processed data is saved into an output CSV file.

    Parameters:
    - url: The URL of the CSV file to fetch.
    - output_file: The name of the output CSV file to save the processed data.
    """
    try:
        # Fetch the CSV file from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Read the CSV data into a DataFrame using io.StringIO
        df = pd.read_csv(io.StringIO(response.text))

        # Ensure that the necessary columns are present
        required_columns = ['first_name', 'second_name', 'now_cost']
        if all(col in df.columns for col in required_columns):
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
            print(f"The required columns {required_columns} are not present in the CSV file.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the CSV file: {e}")
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file: {e}")

def main():
    url = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/cleaned_players.csv'
    output_file = 'player_scripts/current_prices.csv'
    fetch_and_process_csv(url, output_file)

if __name__ == "__main__":
    main()