from scrape import download_understat_csv_files
from clean import clean_understat_data


if __name__ == "__main__": 
    download_understat_csv_files(); 
    clean_understat_data();  