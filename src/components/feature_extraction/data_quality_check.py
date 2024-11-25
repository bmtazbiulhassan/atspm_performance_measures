import pandas as pd
import glob
import yaml
import tqdm
import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.config import FeatureExtractionDirpath, get_relative_base_dirpath
from src.utils import get_root_directory, get_column_name_by_partial_name, get_single_unique_value, create_dict, load_data, export_data
from src.components.feature_extraction.feature_extraction import CoreEventUtils


# Get the root directory of the project
root_dir = get_root_directory()

# Load the YAML configuration file (using absolute path)
with open(os.path.join(root_dir, "config/components", "feature_extraction.yaml"), "r") as file:
    config = yaml.safe_load(file)

# Retrieve settings for data quality check
config = config["data_quality_check"]


# Instantiate class to get directory paths
feature_extraction_dirpath = FeatureExtractionDirpath()

# Get relative base directory path for raw data
_, relative_interim_database_dirpath, relative_production_database_dirpath = get_relative_base_dirpath()


# Instantiate the CoreEventUtils class for event utility methods
core_event_utils = CoreEventUtils()


class DataQualityCheck:
    def __init__(self, event_type: str):
        """
        Initialize the class with the type of event sequence to check.

        Parameters:
        -----------
        event_type : str
            Type of event sequence to check ('vehicle_signal' or 'vehicle_traffic').
        """
        self.event_type = event_type

    def check_data_quality(self, signal_ids: list,day: int, month: int, year: int):
        """
        Perform data quality checks on ATSPM event data by analyzing event sequences.

        Parameters:
        -----------
        signal_ids : list
            List of specific signal IDs to process.
        day : int
            Day of the date (1-31) to filter data by.
        month : int
            Month of the date (1-12) fto filter data by.
        year : int
            Year of the date (e.g., 2024) to filter data by.

        Returns:
        --------
        None: The function saves the preprocessed data as a CSV file in the specified export directory.
        """
        try:
            for signal_id in tqdm.tqdm(signal_ids):
                # Path (from database directory) to directory where sorted event and signal configuration data are stored
                interim_event_dirpath, interim_config_dirpath, _ = feature_extraction_dirpath.get_data_quality_check_dirpath(signal_id=signal_id)

                # Absolute directory path to stored sorted event data
                event_dirpath = os.path.join(root_dir, relative_interim_database_dirpath, interim_event_dirpath)

                if not os.path.isdir(event_dirpath):
                    continue

                try:
                    # Load event data
                    df_event_id = load_data(base_dirpath=os.path.join(root_dir, relative_interim_database_dirpath), 
                                            sub_dirpath= interim_event_dirpath,
                                            filename=f"{year}-{month:02d}-{day:02d}", 
                                            file_type="pkl")

                    # Load signal configuraiton data
                    df_config_id = load_data(base_dirpath=os.path.join(root_dir, relative_interim_database_dirpath),
                                             sub_dirpath=interim_config_dirpath, 
                                             filename=f"{signal_id}",
                                             file_type="csv")
                    
                except FileNotFoundError as e:
                    logging.error(f"File not found: {e}")
                    raise CustomException(custom_message=f"File not found: {e}", sys_module=sys)

                # Filter event data by the specified day
                try:
                    df_event_id = core_event_utils.filter_by_day(df_event=df_event_id, day=day, month=month, year=year)
                except Exception as e:
                    logging.error("Error filtering data by date.")
                    raise CustomException(custom_message=f"Error filtering data by date: {e}", sys_module=sys)

                # Get column names dynamically based on partial matches
                dict_column_names = {
                    "param": get_column_name_by_partial_name(df=df_event_id, partial_name="param"),
                    "code": get_column_name_by_partial_name(df=df_event_id, partial_name="code"),

                    "intersectionType": get_column_name_by_partial_name(df=df_config_id, partial_name="intersection"),
                    "district": get_column_name_by_partial_name(df=df_config_id, partial_name="district"),
                    "county": get_column_name_by_partial_name(df=df_config_id, partial_name="county")
                }

                # Retrieve valid event sequences from configuration
                dict_valid_event_sequence = config["sunstore"]["valid_event_sequence"]

                # Check if the specified event type is valid and retrieve corresponding event sequence
                if self.event_type in dict_valid_event_sequence:
                    valid_event_sequence = dict_valid_event_sequence[self.event_type]
                    unique_params, error_sequence_counter, correct_sequence_counter = self.inspect_sequence(
                        df_event=df_event_id,
                        event_param_column_name=dict_column_names["param"],
                        valid_event_sequence=valid_event_sequence
                    )
                else:
                    logging.error(f"{self.event_type} is not valid.")
                    raise CustomException(
                        custom_message=f"{self.event_type} is not valid. Must be 'vehicle_signal' or 'vehicle_traffic'.", 
                        sys_module=sys
                        )
                
                # Initialize a dictionary to store data quality check information
                int_keys = [
                    'errorSequenceCount', 'correctSequenceCount', 'errorSequencePercent', 'correctSequencePercent'
                ]
                dict_quality_check_id = create_dict(int_keys=int_keys)

                # Update dictionary with error and correct sequence counts and percentages
                total_sequences = error_sequence_counter + correct_sequence_counter

                dict_quality_check_id['errorSequenceCount'] = error_sequence_counter
                dict_quality_check_id['correctSequenceCount'] = correct_sequence_counter
                dict_quality_check_id['errorSequencePercent'] = (error_sequence_counter / (total_sequences + 1)) * 100
                dict_quality_check_id['correctSequencePercent'] = (correct_sequence_counter / (total_sequences + 1)) * 100

                # Convert to DataFrame for easier export
                df_quality_check_id = pd.DataFrame(dict_quality_check_id)

                # Add signal ID and unique parameters to DataFrame
                column_suffix = "phaseNo" if self.event_type == "vehicle_signal" else "channelNo"
                df_quality_check_id[column_suffix] = unique_params

                # Add additional attributes from configuration data
                df_quality_check_id["intersectionType"] = get_single_unique_value(df=df_config_id, column_name=dict_column_names["intersectionType"])
                df_quality_check_id["district"] = get_single_unique_value(df=df_config_id, column_name=dict_column_names["district"])
                df_quality_check_id["county"] = get_single_unique_value(df=df_config_id, column_name=dict_column_names["county"])
                
                # Add date information
                df_quality_check_id["date"] = pd.Timestamp(f"{year}-{month}-{day}").date()

                # Path (from database directory) to directory where sorted event and signal configuration data are stored
                _, _, production_check_dirpath = feature_extraction_dirpath.get_data_quality_check_dirpath(event_type=self.event_type)

                # Save the quality check report as a CSV file in the export directory
                export_data(df=df_quality_check_id, 
                            base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                            sub_dirpath=production_check_dirpath,
                            filename=f"{signal_id}", 
                            file_type="csv")

        except Exception as e:
            logging.error("Error in check_data_quality function")
            raise CustomException(custom_message=f"Error in check_data_quality function: {e}", sys_module=sys)

    def inspect_sequence(self, df_event: pd.DataFrame, event_param_column_name: str, valid_event_sequence: list):
        """
        Inspect sequences in the event data to validate against the expected sequence.

        Parameters:
        -----------
        df_event : pd.DataFrame
            DataFrame containing raw ATSPM event data.
        event_param_column_name : str
            Name of the column to filter by phases or channels.
        valid_event_sequence : list
            Expected sequence of event codes.

        Returns:
        --------
        tuple
            List of unique parameters, error sequence count, and correct sequence count.
        """
        try:
            df_event = core_event_utils.filter_by_event_sequence(df_event=df_event, event_sequence=valid_event_sequence)
            unique_params = sorted(df_event[event_param_column_name].unique())
            error_sequence_counter, correct_sequence_counter = [], []

            for param in unique_params:
                # Filter data by specific phase/channel number
                df_event_param = df_event[df_event[event_param_column_name] == param].reset_index(drop=True)

                # Add sequence ID to identify unique event sequences
                df_event_param["sequenceID"] = core_event_utils.add_event_sequence_id(df_event=df_event_param, 
                                                                                      valid_event_sequence=valid_event_sequence)
                
                # Check sequence validity and count errors and correct sequences
                errors, corrects = self._check_event_sequence(
                    df_event=df_event_param, valid_event_sequence=valid_event_sequence
                )
                error_sequence_counter.append(errors)
                correct_sequence_counter.append(corrects)

            return unique_params, error_sequence_counter, correct_sequence_counter   

        except Exception as e:
            logging.error("Error in inspect_sequence function")
            raise CustomException(custom_message=f"Error in inspect_sequence function: {e}", sys_module=sys)

    def _check_event_sequence(self, df_event: pd.DataFrame, valid_event_sequence: list):
        """
        Validate if event code sequences match the expected sequence for each sequence ID.

        Parameters:
        -----------
        df_event : pd.DataFrame
            DataFrame containing raw ATSPM event data.
        valid_event_sequence : list
            Expected valid sequence of event codes.

        Returns:
        --------
        tuple
            Counts of incorrect and correct sequences.
        """
        try:
            # Dynamically fetch column names
            dict_column_names = {
                "code": get_column_name_by_partial_name(df=df_event, partial_name="code")
            }
            
            # Initialize counters for correct and incorrect sequences
            error_sequence_counter, correct_sequence_counter = 0, 0
            sequence_ids = df_event["sequenceID"].unique()

            for sequence_id in sequence_ids:
                # Extract event codes for the current sequence
                current_sequence = df_event[df_event["sequenceID"] == sequence_id][dict_column_names["code"]].tolist()
                
                # Compare the current sequence with the valid sequence
                if current_sequence == valid_event_sequence:
                    correct_sequence_counter += 1
                else:
                    error_sequence_counter += 1

            return error_sequence_counter, correct_sequence_counter

        except Exception as e:
            logging.error("Error in _check_event_sequence function")
            raise CustomException(custom_message=f"Error in _check_event_sequence function: {e}", 
                                  sys_module=sys)
    





