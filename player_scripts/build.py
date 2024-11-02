import subprocess

def scrape():
    """
    Runs the main methods of the run_understat.py and run_fpl_data.py scripts,
    which are located in the understat and fpl_gw subdirectories respectively.
    """
    try:
        subprocess.run(["python", "understat/run_understat.py"], check=True)
        subprocess.run(["python", "fpl_gw/run_fpl_data.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

def merge(): 
    """
    Runs the main methods to merge the understat data with the fpl_gw data.
    """
    try:
        subprocess.run(["python", "merge_data/merge.py"], check=True)
        subprocess.run(["python", "merge_data/clean.py"], check=True)
        subprocess.run(["python", "merge_data/form.py"], check=True)
        subprocess.run(["python", "merge_data/next_gw.py"], check=True)
        subprocess.run(["python", "merge_data/fixture.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

def train(): 
    try:
        subprocess.run(["python", "train_data/create.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

def prices(): 
    try:
        subprocess.run(["python", "player_prices/create.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

def predict(): 



if __name__ == "__main__":
    scrape()
    merge()
    train()
    prices()