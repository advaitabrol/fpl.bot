import pandas as pd
import numpy as np


def get_top_10_by_column(file_path, column_name, ascending=False):
    """
    Opens a CSV file, sorts the data by a specified column, and returns the top 10 rows for that column.

    Parameters:
    - file_path (str): The path to the CSV file.
    - column_name (str): The column to sort by.
    - ascending (bool): Sort in ascending order if True, descending if False (default: descending).

    Returns:
    - pd.DataFrame: The top 10 rows sorted by the specified column.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check if the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the CSV file.")

        # Sort the DataFrame by the specified column and get the top 10
        sorted_df = df.sort_values(by=column_name, ascending=ascending).head(10)

        return sorted_df

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
top_10 = get_top_10_by_column('prediction_data/2024-25/GW10/fwd.csv', 'predicted_next_week_points', ascending=False)
print(top_10)


def get_top_10_columns_with_ranking(file_path, name, name_column='name'):
    """
    Searches for a row by a specified name in a CSV file, ranks all columns by their values,
    and returns the top 5 columns with both their values and rankings relative to all rows.

    Parameters:
    - file_path (str): The path to the CSV file.
    - name (str): The name to search for.
    - name_column (str): The column in the CSV file that contains names (default is 'name').

    Returns:
    - dict: A dictionary with the top 5 columns, their values, and rankings for the specified name.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check if the name column exists in the DataFrame
        if name_column not in df.columns:
            raise ValueError(f"Column '{name_column}' does not exist in the CSV file.")

        # Check if the specified name exists in the DataFrame
        if name not in df[name_column].values:
            raise ValueError(f"Name '{name}' not found in the '{name_column}' column.")

        # Get the row for the specified name
        target_row = df[df[name_column] == name].iloc[0]

        # Drop non-numeric columns and the name column for ranking purposes
        numeric_df = df.drop(columns=[col for col in df.columns if df[col].dtype == 'object' or col == name_column])

        # Rank each column in descending order (highest values get rank 1)
        ranked_df = numeric_df.rank(ascending=False, method='min')

        # Get the rankings for the target row
        target_rankings = ranked_df.loc[df[name_column] == name].iloc[0]

        # Get the top 5 columns for the target row by ranking
        top_10_columns = target_rankings.nsmallest(10).index
        top_10_values = target_row[top_10_columns]
        top_10_rankings = target_rankings[top_10_columns]

        # Format the result as a dictionary with column names, values, and ranks
        result = {col: {
                    'value': round(top_10_values[col], 2) if np.issubdtype(type(top_10_values[col]), np.number) else top_10_values[col], 
                    'rank': int(top_10_rankings[col])
                 } for col in top_10_columns}

        return result

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
top_10_columns = get_top_10_columns_with_ranking('prediction_data/2024-25/GW10/fwd.csv', 'Erling Haaland')

#print(top_10_columns)