import os
import shutil

def clean_footprint(base_dirs=['fpl_gw_data', 'understat_data', 'gw_data']):
    """
    Deletes all files and directories within the provided base directories.
    
    Parameters:
    - base_dirs: A list of base directories to clean (e.g., ['fpl_gw_data', 'understat_data']).
    """
    for base_dir in base_dirs:
        # Check if the directory exists
        if os.path.exists(base_dir):
            # Loop through all subdirectories and files in the base directory
            for root, dirs, files in os.walk(base_dir, topdown=False):
                # Remove files
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
                    except OSError as e:
                        print(f"Error deleting file {file_path}: {e}")

                # Remove directories
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Deleted directory: {dir_path}")
                    except OSError as e:
                        print(f"Error deleting directory {dir_path}: {e}")

            # Remove the base directory itself
            try:
                shutil.rmtree(base_dir)
                print(f"Deleted base directory: {base_dir}")
            except OSError as e:
                print(f"Error deleting base directory {base_dir}: {e}")
        else:
            print(f"Directory {base_dir} does not exist. Skipping.")

if __name__ == "__main__": 
    clean_footprint(); 