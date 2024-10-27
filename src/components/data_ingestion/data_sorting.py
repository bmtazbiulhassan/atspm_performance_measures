import pandas as pd
import yaml
import tqdm
import sys
import os
import gc

from src.exception import CustomException
from src.logger import logging
from src.utils import get_root_directory, get_column_name_by_partial_name


# Get the root directory of the project
root_dir = get_root_directory()

# Load the YAML configuration file (using absolute path)
with open(os.path.join(root_dir, "config.yaml"), "r") as file:
    config = yaml.safe_load(file)

config = config["data_sorting"]


class DataSort:
    def __init__(self, relative_parent_import_dir: str, relative_export_dir: str):
        """
        Initialize DataSort with directories for raw data and sorted output.

        Parameters:
        -----------
        relative_parent_import_dir : str
            Directory where sub-directories containing raw ATSPM event data CSV files are stored.
        relative_export_dir : str
            Directory for saving sorted data files by signal as pickle files.
        """
        self.relative_parent_import_dir = relative_parent_import_dir
        self.relative_export_dir = relative_export_dir
        self.absolute_export_dir = os.path.join(root_dir, self.relative_export_dir)

    def export(self, df_event: pd.DataFrame, signal_id: str):
        """
        Processes and exports ATSPM data for a specific signal, appending it to existing data if applicable.

        Parameters:
        -----------
        df_event : pd.DataFrame
            DataFrame containing raw ATSPM event data.
        signal_id : str
            Unique identifier for the signal to filter and process.

        Outputs:
        --------
        None: Exports sorted data for each signal as a pickle file.
        """
        try:
            # Filter data for the specified signal ID and reset the index
            df_event_id = df_event[df_event.signalID.astype(str) == signal_id].reset_index(drop=True)

            # Get column names
            dict_column_names = {
                "time": get_column_name_by_partial_name(df=df_event, partial_name="time")
                }

            # Parse and sort timestamps
            df_event_id[dict_column_names["time"]] = pd.to_datetime(df_event_id[dict_column_names["time"]], format="%m-%d-%Y %H:%M:%S.%f")
            df_event_id = df_event_id.sort_values(by=dict_column_names["time"]).reset_index(drop=True)

            # Define the export path for the signal data
            event_filepath = os.path.join(self.absolute_export_dir, f"{signal_id}.pkl")

            # Check if a sorted file already exists for this signal
            if os.path.exists(event_filepath):
                # Append existing data to the current data
                df_event_id = pd.concat([df_event_id, pd.read_pickle(event_filepath)], ignore_index=True)
                df_event_id = df_event_id.drop_duplicates().sort_values(by=dict_column_names["time"]).reset_index(drop=True)

            # Save the data as a pickle file
            df_event_id.to_pickle(event_filepath)

        except Exception as e:
            logging.error(f"Error exporting data for signal ID {signal_id}: {e}")
            raise CustomException(
                custom_message=f"Failed to export data for signal ID {signal_id}", sys_module=sys
                )

    def sort_and_export(self, signal_ids: list, day: int, month: int, year: int):
        """
        Sort, and export ATSPM event data for each signal ID within a specific date.

        Parameters:
        -----------
        signal_ids : list
            List of signal IDs to process. Each ID represents a unique intersection.
        day : int
            Day of the desired date (1-31) for which ATSPM data is sorted.
        month : int
            Month of the desired date (1-12) for which ATSPM data is sorted.
        year : int
            Year of the desired date (e.g., 2024) for which ATSPM data is sorted.

        Outputs:
        --------
        None: Exports sorted data for each signal ID as a pickle file, tracks sorted files in a checker file,
        and logs each step of the process. If any error occurs during data sorting, or exporting, it raises a 
        CustomException with details.
        """
        try:
            # Ensure the export directory exists
            os.makedirs(self.absolute_export_dir, exist_ok=True)

            # Define the path for daily data to process
            absolute_import_dir = os.path.join(self.relative_parent_import_dir, f"{year}-{month:02d}")
            event_filepath = os.path.join(root_dir, absolute_import_dir, f"atspm-{year}-{month}-{day}.csv")

            # Define the path for the check file that tracks sorted files
            check_filepath = os.path.join(self.absolute_export_dir, "checker.csv")

            # Initialize the check file if it doesn't exist
            if not os.path.exists(check_filepath):
                df_check = pd.DataFrame(columns=["fileProc", "isOk"])
                df_check.to_csv(check_filepath, index=False)

            # Load the check file
            df_check = pd.read_csv(check_filepath)

            # Skip files that have already been sorted
            if event_filepath in df_check["fileProc"].unique():
                logging.info(f"File {event_filepath} has already been sorted")
                print(f"File {event_filepath} has already been sorted")
                return

            # Load the event data file
            print(f"Loading: {event_filepath} ...")
            df_event = pd.read_csv(event_filepath, engine="pyarrow")
            print(f"Data Shape: {df_event.shape}")

            # Get column names
            dict_column_names = {
                "signalID": get_column_name_by_partial_name(df=df_event, partial_name="signalID")
                }

            # Filter for the provided signal IDs
            df_event = df_event[df_event[dict_column_names["signalID"]].isin(signal_ids)]

            # Validate the presence of signal IDs
            is_ok = "yes" if len(df_event[dict_column_names["signalID"]].unique()) > 1 else "no"
            if is_ok == "no":
                logging.info(f"File {event_filepath} contains no valid signal IDs")
                raise CustomException(
                    custom_message=f"No valid signal IDs in file {event_filepath}", sys_module=sys
                    )

            # Process each unique signal ID
            for signal_id in tqdm.tqdm(df_event[dict_column_names["signalID"]].unique()):
                logging.info(f"Processing data for signal ID: {signal_id}")
                self.export(df_event, signal_id)

            # Update the check file with the sorted file status
            df_check = pd.concat([df_check, pd.DataFrame({"fileProc": [event_filepath], "isOk": [is_ok]})], ignore_index=True)
            df_check.to_csv(check_filepath, index=False)

            # Clear memory
            gc.collect()

        except Exception as e:
            logging.error(f"Error in sorting for file {event_filepath}: {e}")
            raise CustomException(custom_message="Failed during data sorting", sys_module=sys)





