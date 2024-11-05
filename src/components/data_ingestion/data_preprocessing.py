import pandas as pd
import yaml
import tqdm
import glob
import sys
import os

# from src.exception import CustomException
from src.logger import logging
from src.config import DataIngestionDirpath, get_relative_base_dirpath
from src.utils import get_root_directory, get_column_name_by_partial_name, float_to_int, export_data


# Get the root directory of the project
root_dir = get_root_directory()


# Load the YAML configuration file (using absolute path)
with open(os.path.join(root_dir, "config/components", "data_ingestion.yaml"), "r") as file:
    config = yaml.safe_load(file)

# Retrieve settings for preprocessing
config = config["data_preprocessing"]


# Instantiate class to get directory paths
data_ingestion_dirpath = DataIngestionDirpath()

# Get relative base directory path for raw data
relative_raw_database_dirpath, relative_interim_database_dirpath, _ = get_relative_base_dirpath()


def preprocess_noemi_reports():
    """
    Loads, preprocesses, and exports NOEMI report data for a specified signal.

    Returns:
    --------
    None: The function saves the preprocessed data as a CSV file in the specified export directory.

    Raises:
    -------
    CustomException
        If any issues occur during loading, renaming columns, or exporting data.
    """
    # Path (from database directory) to directory where sorted event data will be exported
    _, interim_config_dirpath = data_ingestion_dirpath.get_data_preprocessing_dirpath()

    # Retrieve the directory name for "intersection" and "lanes" tables from the YAML config
    intersection_dirname = config["noemi"]["intersection_dirname"]
    lanes_dirname = config["noemi"]["lanes_dirname"]

    # Path (from database directory) to directory where "lanes" tables are stored
    raw_report_lanes_dirpath, _ = data_ingestion_dirpath.get_data_preprocessing_dirpath(table_id=lanes_dirname)

    # List all "lanes" filepaths
    lanes_report_filepaths = os.path.join(root_dir, relative_raw_database_dirpath, raw_report_lanes_dirpath, "*.csv")
    lanes_report_filepaths = [filepath for filepath in glob.glob(lanes_report_filepaths)]

    for lane_report_filepath in tqdm.tqdm(lanes_report_filepaths):
        try:
            # Load "lane" report data
            df_lane_report_id = pd.read_csv(lane_report_filepath)

            # lane_filepath: "/media/bm638305/Elements/Codebase/Python/ucfsst_projects/ATSPM/citysignal/data/raw/atspm/fdot_d5/noemi_report/lanes/MRN-0301.csv"

            # Get signal ID
            signal_id = os.path.splitext(os.path.basename(lane_report_filepath))[0]

            # Retrieve column rename map from the YAML configuration
            dict_rename_map = config["noemi"]["rename_map"]

            # Rename columns according to the configuration map
            df_lane_report_id = df_lane_report_id.rename(columns=dict_rename_map)

            # Get a copy of "lane" report
            df_report_id = df_lane_report_id.copy()

            # Path (from database directory) to directory where "intersection" tables are stored
            raw_report_intersection_dirpath, _ = data_ingestion_dirpath.get_data_preprocessing_dirpath(
                table_id=intersection_dirname
            )

            # Path to "intersection" table for the current signal ID
            intersection_report_filepath = os.path.join(
                root_dir, relative_raw_database_dirpath, raw_report_intersection_dirpath, f"{signal_id}.csv"
            )
            
            # Load "intersection" report data
            df_intersection_report_id = pd.read_csv(intersection_report_filepath)

            # Transpose "intersection" report to get columns (by manual review)
            df_intersection_report_id = df_intersection_report_id.set_index(keys="Field").T

            # Get column names in "intersection" report
            dict_column_names = {
                "intersectionType": get_column_name_by_partial_name(df=df_intersection_report_id, partial_name="intersection"),
                "district": get_column_name_by_partial_name(df=df_intersection_report_id, partial_name="district"),
                "county": get_column_name_by_partial_name(df=df_intersection_report_id, partial_name="county")
                }
            
            # Append data from "intersection" report (by manual review)
            df_report_id["intersectionType"] = df_intersection_report_id.loc["Value", dict_column_names["intersectionType"]].lower()
            df_report_id["district"] = df_intersection_report_id.loc["Value", dict_column_names["district"]]
            df_report_id["county"] = df_intersection_report_id.loc["Value", dict_column_names["county"]].lower()

            # Convert float columns to integers where applicable
            df_report_id = float_to_int(df_report_id)

            # Save the preprocessed report as a CSV file in the export directory
            export_data(df=df_report_id, 
                        base_dirpath=os.path.join(root_dir, relative_interim_database_dirpath), 
                        sub_dirpath=interim_config_dirpath,
                        filename=f"{signal_id}", 
                        file_type="csv")
        
        except Exception as e:
            logging.error(f"Error processing NOEMI report for signal ID {signal_id}: {e}")
            # raise CustomException(custom_message="Error during preprocessing of NOEMI report.", sys_module=sys)

