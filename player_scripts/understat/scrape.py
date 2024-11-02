import os
import aiohttp
import asyncio
from aiofiles import open as aio_open

def create_master_folder(master_folder):
    """Creates the master folder if it doesn't exist."""
    if not os.path.exists(master_folder):
        os.makedirs(master_folder)

async def download_file(session, season, file, master_folder):
    file_url = file['download_url']
    file_name = file['name']
    season_folder = os.path.join(master_folder, season)
    if not os.path.exists(season_folder):
        os.makedirs(season_folder)

    file_path = os.path.join(season_folder, file_name)
    async with session.get(file_url) as response:
        if response.status == 200:
            async with aio_open(file_path, 'wb') as f:
                await f.write(await response.read())
            print(f"Downloaded {file_name} for {season}")
        else:
            print(f"Failed to download {file_name} for {season}: {response.status}")

async def fetch_files_for_season(season, api_url, master_folder):
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url) as response:
            if response.status == 200:
                files = await response.json()
                csv_files = [file for file in files if file['name'].endswith('.csv')]
                download_tasks = [download_file(session, season, file, master_folder) for file in csv_files]
                await asyncio.gather(*download_tasks)
            else:
                print(f"Failed to retrieve file list for {season}: {response.status}")

def download_understat_csv_files(base_url='https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/season/understat/', seasons=['2021-22', '2022-23', '2023-24', '2024-25']):
    """
    Downloads all CSV files from the understat GitHub directory for the specified seasons.
    Saves them in a directory in the current working directory.

    Parameters:
    base_url (str): The base URL for the raw GitHub content (e.g., 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/season/understat/').
    seasons (list): A list of seasons (e.g., ['2021-22', '2022-23', '2023-24']).
    """
    master_folder = os.path.join(os.getcwd(), 'understat_data')
    create_master_folder(master_folder)

    async def main():
        tasks = []
        for season in seasons:
            api_url = f'https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data/{season}/understat/'
            tasks.append(fetch_files_for_season(season, api_url, master_folder))
        await asyncio.gather(*tasks)

    asyncio.run(main())

# Example usage
'''
if __name__ == "__main__":
    download_understat_csv_files()
'''