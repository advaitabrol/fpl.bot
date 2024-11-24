import subprocess;
import argparse; 

from understat.run_understat import main as main_understat
from fpl_gw.run_fpl_data import main as main_fpl
from merge_data.run_merge import main as main_merge
from prediction_data.run_predict import main_create_prediction as main_predict
from clean_dirs.remove_files import clean_footprint

def scrape(seasons):
    """
    Runs the main methods of the run_understat.py and run_fpl_data.py scripts,
    which are located in the understat and fpl_gw subdirectories respectively.
    """
    try:
        main_understat(seasons=seasons)
        main_fpl(seasons=seasons)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

def merge(seasons): 
    """
    Runs the main methods to merge the understat data with the fpl_gw data.
    """
    try:
        main_merge(seasons=seasons)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

def train(): 
    try:
        subprocess.run(["python", "player_scripts/train_data/create.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

def prices(): 
    try:
        subprocess.run(["python", "player_scripts/player_prices/create.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

def predict(x): 
    try:
        main_predict(x);
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

def availability():
    try:
        subprocess.run(["python", "player_scripts/player_fitness/availability.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

def clean(): 
    clean_footprint(); 

def main(prediction_folder, status):
    if(status == 'full'):
        all_seasons = ['2021-22', '2022-23', '2023-24', '2024-25']
        """Main function to perform tasks with prediction folder and season arguments."""    
        # Now you can call the necessary functions with these arguments
        scrape(all_seasons)
        merge(all_seasons)
        train()
        prices()
        availability()
        predict(prediction_folder)
        clean()

    if(status == 'partial'): 
        some_seasons = ['2023-24', '2024-25']
        scrape(some_seasons)
        merge(some_seasons)
        prices()
        availability()
        predict(prediction_folder)
        clean(); 

    if(status == 'recalculate'): 
        #prices()
        #availability()
        predict(prediction_folder)
        clean()
    
    if(status == 'clean'): 
        clean(); 


if __name__ == "__main__":
    # Set up argparse to get prediction_folder and season as arguments
    parser = argparse.ArgumentParser(description="Run predictions with specified folder and season.")
    parser.add_argument('prediction_folder', type=str, help="Name of the prediction folder (e.g., 'GW10')")
    parser.add_argument('status', type=str, help="What kind of udpate is this? (full re-train (full), next gw train (partial), or next gw recalculation (recalculate))")

    args = parser.parse_args()

    main(args.prediction_folder, args.status)

