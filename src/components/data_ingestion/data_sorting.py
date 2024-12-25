import pandas as pd
import tqdm
import yaml
import sys
import os
import gc

from src.exception import CustomException
from src.logger import logging
from src.config import DataIngestionDirpath, get_relative_base_dirpath
from src.utils import get_root_directory, get_column_name_by_partial_name, export_data


# Get the root directory of the project
root_dir = get_root_directory()


# Instantiate class to get directory paths
data_ingestion_dirpath = DataIngestionDirpath()

# Get relative base directory path for raw data
relative_raw_database_dirpath, relative_interim_database_dirpath, _ = get_relative_base_dirpath()


class DataSort:
    
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
        None: Exports sorted data for each signal and date as pickle files.
        """
        try:
            # Get column names
            dict_column_names = {
                "time": get_column_name_by_partial_name(df=df_event, partial_name="time"),
                "signalID": get_column_name_by_partial_name(df=df_event, partial_name="signalID")
                }
            
            # Filter data for the specified signal ID and reset the index
            df_event_id = df_event[df_event[dict_column_names["signalID"]].astype(str) == signal_id].reset_index(drop=True)

            # Parse and sort timestamps
            df_event_id[dict_column_names["time"]] = pd.to_datetime(df_event_id[dict_column_names["time"]], 
                                                                    format="%m-%d-%Y %H:%M:%S.%f")
            
            # Adjustment for eastern time zone
            df_event_id[dict_column_names["time"]] = df_event_id[dict_column_names["time"]] + pd.Timedelta(hours=4)
            df_event_id = df_event_id.sort_values(by=dict_column_names["time"]).reset_index(drop=True)
            
            # Path (from database directory) to directory where sorted event data will be exported
            _, interim_event_dirpath = data_ingestion_dirpath.get_data_sorting_dirpath(signal_id=signal_id)

            # Absolute directory path to export sorted event data
            event_dirpath = os.path.join(root_dir, relative_interim_database_dirpath, interim_event_dirpath)

            # Ensure the export directory exists
            os.makedirs(event_dirpath, exist_ok=True)

            # Check if a sorted file already exists for this signal
            if os.path.exists(f"{event_dirpath}/{signal_id}.pkl"):
                # Append existing data to the current data
                df_event_id = pd.concat([df_event_id, pd.read_pickle(f"{event_dirpath}/{signal_id}.pkl")], 
                                        ignore_index=True)
                df_event_id = df_event_id.drop_duplicates().sort_values(by=dict_column_names["time"]).reset_index(drop=True)

            # Convert the timestamp column to datetime if it is not already
            if not pd.api.types.is_datetime64_any_dtype(df_event[dict_column_names["time"]]):
                df_event_id[dict_column_names["time"]] = pd.to_datetime(df_event_id[dict_column_names["time"]], 
                                                                        format="%Y-%m-%d %H:%M:%S.%f")
                
            # Create a new 'date' column containing only the date part (no time)
            df_event_id['date'] = df_event_id[dict_column_names["time"]].dt.date

            # Save pickle files (for each date)
            for date in df_event_id["date"].unique():
                df_event_date = df_event_id[df_event_id["date"] == date].reset_index(drop=True)

                # Export as pickle file
                export_data(df=df_event_date, 
                            base_dirpath=os.path.join(root_dir, relative_interim_database_dirpath), 
                            sub_dirpath=interim_event_dirpath,
                            filename=f"{date.year}-{date.month:02d}-{date.day:02d}", 
                            file_type="pkl")
            
            # Save the data as a pickle file (for each signal)
            export_data(df=df_event_id, 
                        base_dirpath=os.path.join(root_dir, relative_interim_database_dirpath), 
                        sub_dirpath=interim_event_dirpath,
                        filename=f"{signal_id}", 
                        file_type="pkl")

        except Exception as e:
            logging.error(f"Error exporting data for signal ID {signal_id}: {e}")
            raise CustomException(
                custom_message=f"Failed to export data for signal ID {signal_id}", sys_module=sys
                )

    def sort_and_export(self, day: int, month: int, year: int, signal_ids: list):
        """
        Sort, and export ATSPM event data for each signal ID within a specific date.

        Parameters:
        -----------
        signal_ids : list
            List of signal IDs to process. Each ID represents a unique intersection.
        day : int
            Day of the date (1-31) for which ATSPM data is sorted.
        month : int
            Month of the date (1-12) for which ATSPM data is sorted.
        year : int
            Year of the date (e.g., 2024) for which ATSPM data is sorted.

        Outputs:
        --------
        None: Exports sorted data for each signal and date as pickle files., tracks sorted files in a checker 
        file, and logs each step of the process. If any error occurs during data sorting, or exporting, it 
        raises a CustomException with details.
        """
        try:
            # Path (from database directory) to directory where checker file (that tracks sorted files) will be exported
            _, interim_checker_dirpath = data_ingestion_dirpath.get_data_sorting_dirpath()

            # Absolute directory path to export checker file
            checker_dirpath = os.path.join(root_dir, relative_interim_database_dirpath, interim_checker_dirpath)

            # Ensure the check export directory exists
            os.makedirs(checker_dirpath, exist_ok=True)

            # Define the path for the checker file
            checker_filepath = os.path.join(checker_dirpath, "checker.csv")

            # Initialize the checker file if it doesn't exist
            if not os.path.exists(checker_filepath):
                df_checker = pd.DataFrame(columns=["fileProc", "isOk"])
                df_checker.to_csv(checker_filepath, index=False)

            # Load the checker file
            df_checker = pd.read_csv(checker_filepath)

            # Path (from database directory) to directory where scraped ATSPM event data is stored
            raw_event_dirpath, _ = data_ingestion_dirpath.get_data_sorting_dirpath(month=month, year=year)

            event_filepath = os.path.join(root_dir, relative_raw_database_dirpath, raw_event_dirpath, 
                                          f"atspm-{year}-{month}-{day}.csv")

            # Skip files that have already been sorted
            if event_filepath in df_checker["fileProc"].unique():
                logging.info(f"File {event_filepath} has already been sorted")
                print(f"File {event_filepath} has already been sorted")
                return

            # Load the event data file
            print(f"Loading: {event_filepath} ...")
            df_event = pd.read_csv(event_filepath, engine="pyarrow")
            print(f"Data Shape: {df_event.shape}")

            # Get column names
            dict_column_names = {
                "signal": get_column_name_by_partial_name(df=df_event, partial_name="signal")
                }

            # Filter for the provided signal IDs
            df_event = df_event[df_event[dict_column_names["signal"]].isin(signal_ids)]

            # Validate the presence of signal IDs
            is_ok = "yes" if len(df_event[dict_column_names["signal"]].unique()) > 1 else "no"
            if is_ok == "no":
                logging.info(f"File {event_filepath} contains no valid signal IDs")
                print(f"File {event_filepath} contains no valid signal IDs")
            else:
                # Process each unique signal ID
                for signal_id in tqdm.tqdm(df_event[dict_column_names["signal"]].unique()):
                    logging.info(f"Processing data for signal ID: {signal_id}")
                    self.export(df_event, signal_id)

                # Update the checker file with the sorted file status
                df_checker = pd.concat([df_checker, pd.DataFrame({"fileProc": [event_filepath], "isOk": [is_ok]})], ignore_index=True)
                df_checker.to_csv(checker_filepath, index=False)

                # Clear memory
                gc.collect()

        except Exception as e:
            logging.error(f"Error in sorting for file {event_filepath}: {e}")
            raise CustomException(custom_message="Failed during data sorting", sys_module=sys)





