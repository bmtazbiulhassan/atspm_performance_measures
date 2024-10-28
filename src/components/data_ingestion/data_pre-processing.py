import pandas as pd
import yaml
import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import get_root_directory, float_to_int


# Get the root directory of the project
root_dir = get_root_directory()

# Load the YAML configuration file (using absolute path)
with open(os.path.join(root_dir, "config.yaml"), "r") as file:
    config = yaml.safe_load(file)

config = config["data_pre-processing"]


def preprocess_noemi_reports(signal_id: str, relative_parent_import_dir: str, relative_export_dir: str):
    """
    Loads, preprocesses, and exports NOEMI report data for a specified signal.

    Parameters:
    -----------
    signal_id : str
        The unique identifier for the signal being processed.
    relative_parent_import_dir : str
        Relative path to the parent directory containing subdirectories with NOEMI report data files.
    relative_export_dir : str
        Relative path to the directory where preprocessed data will be saved.

    Returns:
    --------
    None: The function saves the preprocessed data as a CSV file in the specified export directory.

    Raises:
    -------
    CustomException
        If any issues occur during loading, renaming columns, or exporting data.
    """
    try:
        # Define absolute paths for import directories
        absolute_parent_import_dir = os.path.join(root_dir, relative_parent_import_dir)
        
        # Retrieve the directory name for "lanes" reports from the YAML config
        report_dir = config["noemi"]["report_dir"]
        absolute_import_dir = os.path.join(absolute_parent_import_dir, report_dir)

        # Load "lanes" report data for the specified signal ID
        df_report = pd.read_csv(f"{absolute_import_dir}/{signal_id}.csv")
        
        # Retrieve column rename map from the YAML configuration
        dict_rename_map = config["noemi"]["rename_map"]

        # Rename columns according to the configuration map
        df_report = df_report.rename(columns=dict_rename_map)

        # Convert float columns to integers where applicable
        df_report = float_to_int(df_report)

        # Ensure the export directory exists
        os.makedirs(relative_export_dir, exist_ok=True)

        # Save the preprocessed report as a CSV file in the export directory
        df_report.to_csv(f"{relative_export_dir}/{signal_id}.csv", index=False)
    
    except FileNotFoundError as e:
        logging.error(f"File for signal ID {signal_id} not found in {absolute_import_dir}: {e}")
        raise CustomException(custom_message=f"File for signal ID {signal_id} not found.", sys_module=sys)
    
    except KeyError as e:
        logging.error(f"Missing configuration for column renaming: {e}")
        raise CustomException(custom_message="Column rename configuration missing in YAML.", sys_module=sys)
    
    except Exception as e:
        logging.error(f"Error processing NOEMI report for signal ID {signal_id}: {e}")
        raise CustomException(custom_message="Error during preprocessing of NOEMI report.", sys_module=sys)

