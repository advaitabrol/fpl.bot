from app.services.player_scripts.merge_data.merge import merge_fpl_and_understat_data
from app.services.player_scripts.merge_data.clean import process_and_sort_data
from app.services.player_scripts.merge_data.form import process_player_data
from app.services.player_scripts.merge_data.next_gw import add_next_gw_points
from app.services.player_scripts.merge_data.fixture import process_fixtures_default

def main(seasons): 
  merge_fpl_and_understat_data(seasons=seasons)
  process_and_sort_data(seasons=seasons) 
  process_player_data(seasons=seasons)
  add_next_gw_points(seasons=seasons)
  process_fixtures_default(seasons=seasons)

