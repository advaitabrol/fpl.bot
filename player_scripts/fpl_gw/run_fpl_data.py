import asyncio

from scrape import download_fpl_gw_csv_files
from reformat import fpl_gw_to_player
from clean import clean_fpl_gw_data


if __name__ == "__main__": 
    asyncio.run(download_fpl_gw_csv_files())
    fpl_gw_to_player(); 
    clean_fpl_gw_data(); 