import os
import yaml

from src.utils import get_root_directory


# Get the root directory
root_dirpath = get_root_directory()

# Load the YAML configuration file (using absolute path)
with open(os.path.join(root_dirpath, "config/general_settings.yaml"), "r") as file:
    config = yaml.safe_load(file)


class DataIngestionDirpath:
    def __init__(self, district):
        self.district = district

    def get_data_scraping_dirpath(self, table_id: str, month: int, year: int):
        
        # Path (from database directory) to directory containing tables scraped from NOEMI reports.
        raw_report_dirpath = config["raw_report_dir"]
        raw_report_dirpath = raw_report_dirpath.format(district=self.district, table_id=table_id)

        # Path (from database directory) to directory containing ATSPM event data scraped from Sunstore
        raw_event_dirpath = config["raw_event_dir"]
        raw_event_dirpath = raw_event_dirpath.format(district=self.district, 
                                                     year=f"{year}", month=f"{month:02d}")
    
        return raw_report_dirpath, raw_event_dirpath
    
    
    
