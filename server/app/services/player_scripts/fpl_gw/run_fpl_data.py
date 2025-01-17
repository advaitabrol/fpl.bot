import asyncio

from app.services.player_scripts.fpl_gw.scrape import download_fpl_gw_csv_files
from app.services.player_scripts.fpl_gw.reformat import fpl_gw_to_player
from app.services.player_scripts.fpl_gw.clean import clean_fpl_gw_data


def main(seasons): 
    asyncio.run(download_fpl_gw_csv_files(seasons=seasons))
    fpl_gw_to_player(); 
    clean_fpl_gw_data(); 