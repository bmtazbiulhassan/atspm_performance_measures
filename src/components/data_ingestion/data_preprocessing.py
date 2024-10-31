import pandas as pd
import yaml
import tqdm
import glob
import sys
import os

# from src.exception import CustomException
from src.logger import logging
from src.utils import get_root_directory, get_column_name_by_partial_name, float_to_int, export_data


# Get the root directory of the project
root_dir = get_root_directory()

# Load the YAML configuration file (using absolute path)
with open(os.path.join(root_dir, "config.yaml"), "r") as file:
    config = yaml.safe_load(file)

# Retrive relative base dir
relative_raw_base_dir = config["relative_base_dir"]["raw"]
relative_interim_base_dir = config["relative_base_dir"]["interim"]

# Retrieve settings for preprocessing
config = config["data_preprocessing"]


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
    # Retrieve report import and config export sub-directories
    report_import_sub_dirs = config["noemi"]["report_import_sub_dirs"]
    config_export_sub_dirs = config["noemi"]["config_export_sub_dirs"]
    
    # Retrieve the directory name for "intersection" and "lanes" tables from the YAML config
    intersection_dir = config["noemi"]["intersection_dir"]
    lanes_dir = config["noemi"]["lanes_dir"]

    # List all "lanes" filepaths
    lanes_filepaths = os.path.join(root_dir, relative_raw_base_dir, *report_import_sub_dirs, lanes_dir, "*.csv")
    lanes_filepaths = [filepath for filepath in glob.glob(lanes_filepaths)]

    for lane_filepath in tqdm.tqdm(lanes_filepaths):
        try:
            # Load "lane" report data
            df_lane_report_id = pd.read_csv(lane_filepath)

            # lane_filepath: "/media/bm638305/Elements/Codebase/Python/ucfsst_projects/ATSPM/citysignal/data/raw/atspm/fdot_d5/noemi_report/lanes/MRN-0301.csv"

            # Get signal ID
            signal_id = os.path.splitext(os.path.basename(lane_filepath))[0]

            # Retrieve column rename map from the YAML configuration
            dict_rename_map = config["noemi"]["rename_map"]

            # Rename columns according to the configuration map
            df_lane_report_id = df_lane_report_id.rename(columns=dict_rename_map)

            # Get a copy of "lane" report
            df_report_id = df_lane_report_id.copy()

            # Filepath to "intersection" report for the given signal ID
            intersection_filepath = os.path.join(root_dir, relative_raw_base_dir, *report_import_sub_dirs, intersection_dir, f"{signal_id}.csv")
            
            # Load "intersection" report data
            df_intersection_report_id = pd.read_csv(intersection_filepath)

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
                        base_dir=os.path.join(root_dir, relative_interim_base_dir), 
                        filename=f"{signal_id}", 
                        file_type="csv", 
                        sub_dirs=config_export_sub_dirs)
        
        except Exception as e:
            logging.error(f"Error processing NOEMI report for signal ID {signal_id}: {e}")
            # raise CustomException(custom_message="Error during preprocessing of NOEMI report.", sys_module=sys)

