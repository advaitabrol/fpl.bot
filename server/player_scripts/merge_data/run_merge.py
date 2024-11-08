from merge_data.merge import merge_fpl_and_understat_data
from merge_data.clean import process_and_sort_data
from merge_data.form import process_player_data
from merge_data.next_gw import add_next_gw_points
from merge_data.fixture import process_fixtures_default

def main(seasons): 
  merge_fpl_and_understat_data(seasons=seasons)
  process_and_sort_data(seasons=seasons) 
  process_player_data(seasons=seasons)
  add_next_gw_points(seasons=seasons)
  process_fixtures_default(seasons=seasons)

