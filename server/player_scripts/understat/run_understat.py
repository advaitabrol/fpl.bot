from understat.scrape import download_understat_csv_files
from understat.clean import clean_understat_data


def main(seasons): 
    download_understat_csv_files(seasons=seasons); 
    clean_understat_data();  