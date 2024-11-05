import subprocess

def scrape():
    """
    Runs the main methods of the run_understat.py and run_fpl_data.py scripts,
    which are located in the understat and fpl_gw subdirectories respectively.
    """
    try:
        subprocess.run(["python", "player_scripts/understat/run_understat.py"], check=True)
        subprocess.run(["python", "player_scripts/fpl_gw/run_fpl_data.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

def merge(): 
    """
    Runs the main methods to merge the understat data with the fpl_gw data.
    """
    try:
        subprocess.run(["python", "player_scripts/merge_data/merge.py"], check=True)
        subprocess.run(["python", "player_scripts/merge_data/clean.py"], check=True)
        subprocess.run(["python", "player_scripts/merge_data/form.py"], check=True)
        subprocess.run(["python", "player_scripts/merge_data/next_gw.py"], check=True)
        subprocess.run(["python", "player_scripts/merge_data/fixture.py"], check=True)
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
        subprocess.run(["python", "player_scripts/prediction_data/create.py {x}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")


if __name__ == "__main__":
    prediction_folder = 'GW10'
    #scrape()
    #merge()
    #train()
    #prices()
    predict(prediction_folder)
