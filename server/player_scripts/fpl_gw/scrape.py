import os
import requests
import asyncio
import aiohttp
from aiofiles import open as aio_open

#SCRAPING FPL_GW DATA --> IN GW FORMAT 
async def download_gw_file(session, url, filepath):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                async with aio_open(filepath, 'wb') as f:
                    await f.write(await response.read())
                print(f"Downloaded {os.path.basename(filepath)}")
            else:
                print(f"Failed to download {os.path.basename(filepath)}: {response.status} (URL: {url})")
    except Exception as e:
        print(f"Error downloading {os.path.basename(filepath)}: {e}")

async def download_fpl_gw_csv_files(
    base_url='https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/',
    seasons=['2021-22', '2022-23', '2023-24', '2024-25'],
    gw_range=range(1, 39),
    master_folder=None):
    """
    Downloads gameweek CSV files from the raw GitHub URL for specified seasons and gameweeks,
    and saves them in a specified directory.

    Parameters:
    base_url (str): The base URL for the raw GitHub content.
    seasons (list): A list of seasons (e.g., ['2021-22', '2022-23', '2023-24']).
    gw_range (range): The range of gameweeks to download (e.g., range(1, 39)).
    master_folder (str): The directory to save the downloaded files. Defaults to an environment variable or 'gw_data'.
    """
    # Set master folder to an environment variable if provided, or use a default path
    if master_folder is None:
        master_folder = os.getenv('FPL_DATA_DIR', 'gw_data')

    # Ensure the master folder exists
    if not os.path.exists(master_folder):
        os.makedirs(master_folder)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for season in seasons:
            # Create season folder if it doesn't exist
            season_folder = os.path.join(master_folder, season)
            if not os.path.exists(season_folder):
                os.makedirs(season_folder)

            for gw in gw_range:
                gw_file = f'gw{gw}.csv'
                gw_url = f'{base_url}{season}/gws/{gw_file}'
                filepath = os.path.join(season_folder, gw_file)
                
                # Add download task to the list
                tasks.append(download_gw_file(session, gw_url, filepath))
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)

# Run the function in an event loop (suitable for server usage)
'''
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Optionally pass a custom directory through an environment variable
    asyncio.run(download_fpl_gw_csv_files())
'''