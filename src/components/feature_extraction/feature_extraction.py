import pandas as pd
import numpy as np
import yaml
import tqdm
import sys
import os
from collections import Counter

from src.exception import CustomException
from src.logger import logging
from src.config import FeatureExtractionDirpath, get_relative_base_dirpath
from src.utils import get_root_directory, get_column_name_by_partial_name, float_to_int, load_data, export_data


# Get the root directory of the project
root_dir = get_root_directory()


# Load the YAML configuration file (using absolute path)
with open(os.path.join(root_dir, "config/components", "feature_extraction.yaml"), "r") as file:
    config = yaml.safe_load(file)

# Retrieve settings for data quality check
config = config["feature_extraction"]


# Instantiate class to get directory paths
feature_extraction_dirpath = FeatureExtractionDirpath()

# Get relative base directory path for raw data
_, relative_interim_database_dirpath, relative_production_database_dirpath = get_relative_base_dirpath()


class CoreEventUtils:

    # def mapping_event_sequence(self, df_event: pd.DataFrame, dict_map):
    #     pass
    
    def filter_by_event_sequence(self, df_event: pd.DataFrame, event_sequence: list):
        """
        Filter a DataFrame based on a specific sequence of event codes, then sort and return it.

        Parameters:
        -----------
        df_event : pd.DataFrame
            DataFrame containing raw ATSPM event data.
        event_sequence : list
            Sequence of event codes to filter (e.g., [1, 8, 10, 11], [21, 22, 23], [82, 81], etc.).

        Returns:
        --------
        pd.DataFrame
            Filtered and sorted DataFrame based on the provided event sequence.

        Raises:
        -------
        CustomException
            If specified columns are not found in the DataFrame.
        """       
        # Get column names
        dict_column_names = {
            "code": get_column_name_by_partial_name(df=df_event, partial_name="code"),
            "time": get_column_name_by_partial_name(df=df_event, partial_name="time")
            }
        
        df_event[dict_column_names["code"]] = df_event[dict_column_names["code"]].astype(int)
        
        # Filter rows based on the specified event sequence
        df_event_sequence = df_event[df_event[dict_column_names["code"]].isin(event_sequence)]
        
        # Sort by timestamp and event code to maintain order
        df_event_sequence = df_event_sequence.sort_values(by=[dict_column_names["time"], dict_column_names["code"]]).reset_index(drop=True)
        
        return df_event_sequence

    def add_event_sequence_id(self, df_event: pd.DataFrame, valid_event_sequence: list):
        """
        Add a 'sequenceID' column to the DataFrame based on consecutive matching sequences of event codes.

        Parameters:
        -----------
        df_event : pd.DataFrame
            DataFrame containing raw ATSPM event data.
        valid_event_sequence : list
            Sequence of event codes that defines a valid sequence (e.g., [1, 8, 10, 11], [82, 81]).

        Returns:
        --------
        sequence_ids: list
            list of sequence ids.

        Raises:
        -------
        CustomException
            If the specified event_code_column or timestamp column is not available in the DataFrame.
        """
        # Get column names
        dict_column_names = {
            "code": get_column_name_by_partial_name(df=df_event, partial_name="code"),
            "time": get_column_name_by_partial_name(df=df_event, partial_name="time")
            }
        
        try:
            # Sort DataFrame by timestamp to ensure chronological ordering
            df_event = df_event.sort_values(by=dict_column_names["time"]).reset_index(drop=True)

            # Initialize sequence ID and tracking variables
            sequence_id = 1
            sequence_ids = []
            current_event_sequence = []

            # Iterate through each row to assign sequence IDs
            for _, row in df_event.iterrows():
                event_code = row[dict_column_names["code"]]

                # Check if appending the current event code maintains a valid sequence
                if not current_event_sequence or self._is_valid_sequence(
                    current_event_sequence=current_event_sequence + [event_code], valid_event_sequence=valid_event_sequence):
                    current_event_sequence.append(event_code)
                else:
                    # If sequence is broken, increment sequence ID and start a new sequence
                    sequence_id += 1
                    current_event_sequence = [event_code]

                # If current sequence does not match the valid sequence, start a new sequence
                if not self._is_valid_sequence(current_event_sequence, valid_event_sequence):
                    sequence_id += 1
                    current_event_sequence = [event_code]

                # Append current sequence ID to the list for each row
                sequence_ids.append(sequence_id)

            return sequence_ids
        
        except Exception as e:
            logging.error("Error adding sequence ID")
            raise CustomException(custom_message=f"Error adding sequence ID: {e}", sys_module=sys)

    def _is_valid_sequence(self, current_event_sequence: list, valid_event_sequence: list) -> bool:
        """
        Check if the current sequence matches the beginning of the specified valid sequence.

        Parameters:
        -----------
        current_event_sequence : list
            The current sequence being evaluated.
        valid_event_sequence : list
            The reference valid sequence of event codes.

        Returns:
        --------
        bool
            True if current_event_sequence matches the start of valid_event_sequence, otherwise False.
        """
        return current_event_sequence == valid_event_sequence[:len(current_event_sequence)]
    
    def filter_by_day(self, df_event: pd.DataFrame, day: int, month: int, year: int):
        """
        Filters a DataFrame to include only rows with timestamps within a specified day.

        Parameters:
        -----------
        df_event : pd.DataFrame
            DataFrame containing raw ATSPM event data.
        day : int
            Day of the desired date (1-31).
        month : int
            Month of the desired date (1-12).
        year : int
            Year of the desired date (e.g., 2024).

        Returns:
        --------
        pd.DataFrame
            DataFrame filtered to include only rows with timestamps within the specified day.

        Raises:
        -------
        CustomException
            If there is an error in date conversion or filtering.
        """
        # Retrieve the timestamp column name dynamically
        dict_column_names = {
            "time": get_column_name_by_partial_name(df=df_event, partial_name="time")
        }
        
        try:
            # Define the start and end of the specified day
            start_date = pd.to_datetime(f"{year}-{month:02d}-{day:02d} 00:00:00")
            end_date = pd.to_datetime(f"{year}-{month:02d}-{day:02d} 23:59:59")
            
            # Convert the timestamp column to datetime if it is not already
            if not pd.api.types.is_datetime64_any_dtype(df_event[dict_column_names["time"]]):
                df_event[dict_column_names["time"]] = pd.to_datetime(df_event[dict_column_names["time"]], 
                                                                     format="%Y-%m-%d %H:%M:%S.%f")
            
            # Filter the DataFrame for rows within the specified date range
            df_event = df_event[((df_event[dict_column_names["time"]] >= start_date) &
                                 (df_event[dict_column_names["time"]] <= end_date))]
            
            return df_event
        
        except Exception as e:
            raise CustomException(custom_message=f"Error filtering by day: {e}", sys_module=sys)
        
    def calculate_hourly_aggregates(self, df: pd.DataFrame, columns: list, only_sum: bool = False):
        """
        Calculate hourly aggregates for the given DataFrame. The function can either calculate
        only the sum or multiple statistics (min, max, mean, std).

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing data to be aggregated.
        columns: list
            List of columns to process.
        only_sum : bool, optional
            If True, calculates only the sum for each column. Otherwise, calculates
            min, max, mean, and standard deviation. Default is False.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing hourly aggregated data.

        Raises:
        -------
        CustomException
            If an error occurs during the aggregation process.
        """
        try:
            logging.info("Starting hourly aggregation process.")

            # Convert "cycleBegin" to datetime and extract the hour
            df["cycleBegin"] = pd.to_datetime(df["cycleBegin"])
            df["hour"] = df["cycleBegin"].dt.hour

            # Drop unnecessary columns
            df = df.drop(columns=["cycleNo", "cycleBegin", "cycleEnd"])

            if only_sum:
                # Calculate only the sum if `only_sum` is True
                df_hourly = df.groupby(["signalID", "date", "hour"])[columns].sum().reset_index()

                # Rename columns to append 'Sum' at the end
                df_hourly.columns = [
                    col + "Sum" if col not in ["signalID", "date", "hour"] else col 
                    for col in df_hourly.columns
                ]

                return df_hourly
            else:
                # Replace 0 values with NaN to avoid bias in statistical calculations
                df[columns] = df[columns].replace(0, np.nan)

                # Define aggregation functions
                agg_funcs = {col: ["min", "max", "mean", "std"] for col in columns}

                # Group by signalID, date, and hour, and apply multiple aggregations
                df_hourly = df.groupby(["signalID", "date", "hour"]).agg(agg_funcs).round(4)

                # Flatten multi-level column names and rename with appropriate suffixes
                df_hourly.columns = [
                    f"{col}{'Avg' if func == 'mean' else func.capitalize()}" 
                    for col, func in df_hourly.columns
                ]

                # Reset index for the final output
                df_hourly = df_hourly.reset_index()

                return df_hourly

        except Exception as e:
            logging.error(f"Error occurred during hourly aggregation: {e}")
            raise CustomException(
                custom_message="Failed to calculate hourly aggregates.",
                sys_module=sys
            )


class TrafficSignalProfile(CoreEventUtils):
    def __init__(self, day: int, month: int, year: int):
        """
        Initialize the class with day, month, and year of the date to process.

        Parameters:
        -----------
        day : int
            Day of the desired date (1-31).
        month : int
            Month of the desired date (1-12).
        year : int
            Year of the desired date (e.g., 2024).
        """
        self.day = day; self.month = month; self.year = year


    def add_barrier_no(self, signal_id: str):
        """
        Adds barrier numbers to the configuration data for a signal.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the signal.

        Returns:
        --------
        pd.DataFrame
            Configuration DataFrame with barrier numbers.
        """
        # Path (from database directory) to directory where signal configuration data is stored
        _, interim_config_dirpath, _, _ = feature_extraction_dirpath.get_feature_extraction_dirpath()

        # Load configuration data for the specific signal
        df_config_id = load_data(base_dirpath=os.path.join(root_dir, relative_interim_database_dirpath), 
                                 sub_dirpath=interim_config_dirpath,
                                 filename=f"{signal_id}",
                                 file_type="csv")

        # Convert columns with float values to integer type where applicable
        df_config_id = float_to_int(df_config_id)

        # Fetch the phase column name based on partial match
        dict_column_names = {
            "phase": get_column_name_by_partial_name(df=df_config_id, partial_name="phase")
            }
        phase_nos = df_config_id[dict_column_names["phase"]].unique().tolist()

        # Check if phase numbers in configuration data have corresponding entries in barrier map
        if all(phase_no not in config["noemi"]["barrier_map"].keys() for phase_no in phase_nos):
            raise CustomException(
                custom_message=f"Barrier map {config['noemi']['barrier_map']} is not valid for signal ID {signal_id}", 
                sys_module=sys
            )

        # Drop rows with missing phase information
        df_config_id = df_config_id.dropna(subset=dict_column_names["phase"])

        try:
            # Map phase numbers to their respective barrier numbers using the barrier map
            df_config_id["barrierNo"] = df_config_id[dict_column_names["phase"]].map(config["noemi"]["barrier_map"])
        except Exception as e:
            logging.error(f"Error in getting barrier no for {signal_id}")
            raise CustomException(custom_message=f"Error {e} in getting barrier no for {signal_id}", 
                                  sys_module=sys)
        
        return df_config_id

    def extract_phase_profile(self, signal_id: str, event_type: str):
        """
        Load, prepare, and extract phase data for a specific event type ('vehicle_signal' or 'pedestrian_signal').

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the signal.
        event_type : str
            Type of event to process - 'vehicle_signal' or 'pedestrian_signal'.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing extracted phase data.
        """
        # Set event sequence and mapping based on signal type
        if event_type == "vehicle_signal":
            # Mapping event codes to respective phase times
            dict_event_code_map = {
                "greenBegin": 1, "greenEnd": 8,
                "yellowBegin": 8, "yellowEnd": 10,
                "redClearanceBegin": 10, "redClearanceEnd": 11
            }
            valid_event_sequence = config["sunstore"]["valid_event_sequence"]["vehicle_signal"]

        elif event_type == "pedestrian_signal":
            # Mapping event codes for pedestrian phases
            dict_event_code_map = {
                "pedestrianWalkBegin": 21, "pedestrianWalkEnd": 22,
                "pedestrianClearanceBegin": 22, "pedestrianClearanceEnd": 23,
                "pedestrianDontWalkBegin": 23
            }
            valid_event_sequence = config["sunstore"]["valid_event_sequence"]["pedestrian_signal"]

        else:
            logging.error(f"{event_type} is not valid.")
            raise CustomException(
                custom_message=f"{event_type} is not valid. Must be 'vehicle' or 'pedestrian'.", 
                sys_module=sys
            )

        try:
            # Load and prepare configuration data with barrier numbers if applicable
            df_config_id = self.add_barrier_no(signal_id=signal_id)

            # Path (from database directory) to directory where sorted event data is stored
            interim_event_dirpath, _, _, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(signal_id=signal_id)

            # Load event data
            df_event_id = load_data(base_dirpath=os.path.join(root_dir, relative_interim_database_dirpath),
                                    sub_dirpath=interim_event_dirpath, 
                                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                    file_type="pkl")

            # Get column names
            dict_column_names = {
                "phase": get_column_name_by_partial_name(df=df_config_id, partial_name="phase"),
                
                "param": get_column_name_by_partial_name(df=df_event_id, partial_name="param"),
                "code": get_column_name_by_partial_name(df=df_event_id, partial_name="code"),
                "time": get_column_name_by_partial_name(df=df_event_id, partial_name="time")
            }

            # Convert the timestamp column to datetime and sort data by time
            df_event_id[dict_column_names["time"]] = pd.to_datetime(df_event_id[dict_column_names["time"]], 
                                                                    format="%Y-%m-%d %H:%M:%S.%f")
            df_event_id = df_event_id.sort_values(by=dict_column_names["time"]).reset_index(drop=True)

            # Filter data based on the valid event sequence for this phase type
            df_event_id = self.filter_by_event_sequence(df_event=df_event_id, 
                                                        event_sequence=valid_event_sequence)

            # Initialize an empty list to store phase profiles
            phase_profile_id = []
            for phase_no in sorted(df_event_id[dict_column_names["param"]].unique()):
                # Filter data for each phase number
                df_event_phase = df_event_id[df_event_id[dict_column_names["param"]] == phase_no]

                # Assign sequence IDs
                df_event_phase = df_event_phase.copy()
                df_event_phase["sequenceID"] = self.add_event_sequence_id(df_event_phase, 
                                                                          valid_event_sequence=valid_event_sequence)

                for sequence_id in df_event_phase["sequenceID"].unique():
                    df_event_sequence = df_event_phase[df_event_phase["sequenceID"] == sequence_id]
                    
                    current_event_sequence = df_event_sequence[dict_column_names["code"]].tolist()
                    correct_flag = int(current_event_sequence == valid_event_sequence)

                    # Generate phase information dictionary, dynamically adding timestamps for each event
                    dict_phase_profile_id = {
                        "signalID": signal_id,
                        "phaseNo": phase_no,
                        "correctSequenceFlag": correct_flag,
                        **{key: df_event_sequence[df_event_sequence[dict_column_names["code"]] == code][dict_column_names["time"]].iloc[0]
                           if code in current_event_sequence else pd.NaT for key, code in dict_event_code_map.items()}
                    }
                    
                    # Conditionally add "barrierNo" if the signal type is "vehicle"
                    if event_type == "vehicle_signal":
                        dict_phase_profile_id["barrierNo"] = config["noemi"]["barrier_map"].get(int(phase_no), 0)

                    phase_profile_id.append(dict_phase_profile_id)

            # Convert phase profile to DataFrame 
            df_phase_profile_id = pd.DataFrame(phase_profile_id)

            # Create a pseudo timestamp for sorting
            time_columns = [column for column in df_phase_profile_id.columns if column.endswith("Begin") or column.endswith("End")]
            
            if len(df_phase_profile_id) != 0:
                df_phase_profile_id["pseudoTimeStamp"] = df_phase_profile_id[time_columns].bfill(axis=1).iloc[:, 0]
                df_phase_profile_id = df_phase_profile_id.sort_values(by="pseudoTimeStamp").reset_index(drop=True)
                df_phase_profile_id.drop(columns=["pseudoTimeStamp"], inplace=True)

                # Add date information
                df_phase_profile_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

            # Path (from database directory) to directory where vehicle and pedestrian signal profile will be stored
            _, _, production_signal_dirpath, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="phase",
                event_type=event_type,
                signal_id=signal_id
            )

            # Save the phase profile data as a pkl file in the export directory
            export_data(df=df_phase_profile_id, 
                        base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                        sub_dirpath=production_signal_dirpath,
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                        file_type="pkl")

            return df_phase_profile_id

        except Exception as e:
            logging.error(f"Error extracting phase profile for signal ID {signal_id}: {e}")
            raise CustomException(
                custom_message=f"Error extracting phase profile for signal ID {signal_id}: {e}", 
                sys_module=sys
                )

    def extract_vehicle_phase_profile(self, signal_id: str):
        """
        Extract vehicle phase data for a specific signal.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the signal.

        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted vehicle phase data.
        """
        return self.extract_phase_profile(signal_id, event_type="vehicle_signal")

    def extract_pedestrian_phase_profile(self, signal_id: str):
        """
        Extract pedestrian phase data for a specific signal.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the signal.

        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted pedestrian phase data.
        """
        return self.extract_phase_profile(signal_id, event_type="pedestrian_signal")

    def assign_cycle_nos(self, df_vehicle_phase_profile: pd.DataFrame, start_barrier_no: int = 1):
        """
        Dynamically assign unique cycle numbers to DataFrame based on specified barrier value.

        Parameters:
        -----------
        df_vehicle_phase_profile : pd.DataFrame
            DataFrame containing extracted vehicle phase profile data.
        start_barrier_no : int, optional
            Barrier number that initiates a new cycle. Default is 1. (Must be either 1 or 2)

        Returns:
        --------
        pd.DataFrame
            The input DataFrame with an added 'cycleNo' column.
        """
        try:
            cycle_no = 0  # Initialize cycle counter
            cycle_nos = []  # List to store cycle numbers for each row

            for idx, row in df_vehicle_phase_profile.iterrows():
                current_barrier_no = row["barrierNo"]
                # Increment cycle number when the start barrier is encountered and is not consecutive
                if (current_barrier_no == start_barrier_no) and (idx == 0 or df_vehicle_phase_profile.loc[idx - 1, "barrierNo"] != start_barrier_no):
                    cycle_no += 1
                cycle_nos.append(cycle_no)  # Append current cycle number

            df_vehicle_phase_profile["cycleNo"] = cycle_nos  # Add cycle numbers to DataFrame

            return df_vehicle_phase_profile

        except Exception as e:
            logging.error("Error assigning cycle numbers")
            raise CustomException(custom_message=f"Error assigning cycle numbers: {e}", sys_module=sys)

    def extract_vehicle_cycle_profile(self, signal_id: str, start_barrier_no: int = 1):
        """
        Extracts the vehicle cycle profile for a given signal ID with dynamic handling for different signal types.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the signal.
        start_barrier_no : int, optional
            Barrier number to start a new cycle. Default is 1. (Must be either 1 or 2)

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the vehicle cycle profile.
        """
        try:
            # Path (from database directory) to directory where phase-level vehicle signal profile is stored
            _, _, production_signal_dirpath, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="phase",
                event_type="vehicle_signal",
                signal_id=signal_id
            )

            # Load vehicle phase profile data 
            df_vehicle_phase_profile_id = load_data(base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                                                    sub_dirpath=production_signal_dirpath,
                                                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                                    file_type="pkl")

            df_vehicle_cycle_profile_id = pd.DataFrame()  # Initialize DataFrame for cycle profiles

            if len(df_vehicle_phase_profile_id) != 0:
                # Assign cycle numbers to the phase profile based on the specified start barrier
                df_vehicle_phase_profile_id = self.assign_cycle_nos(df_vehicle_phase_profile=df_vehicle_phase_profile_id, start_barrier_no=start_barrier_no)
                cycle_nos = sorted(df_vehicle_phase_profile_id["cycleNo"].unique())  # Get sorted unique cycle numbers

                # Process each cycle number to collect phase and cycle times
                for cycle_no in cycle_nos:
                    # Filter data for the current cycle and reset index
                    df_vehicle_phase_profile_cycle = df_vehicle_phase_profile_id[df_vehicle_phase_profile_id["cycleNo"] == cycle_no].reset_index(drop=True)
                    
                    # Skip the cycle if it has incorrect sequence flags
                    if 0 in df_vehicle_phase_profile_cycle["correctSequenceFlag"].unique():
                        continue
                    
                    cycle_begin = df_vehicle_phase_profile_cycle.iloc[0]["greenBegin"]
                    cycle_end = df_vehicle_phase_profile_cycle.iloc[-1]["redClearanceEnd"]
                    
                    # Initialize dictionary to store cycle information for the current cycle
                    dict_vehicle_cycle_profile_id = {
                        "signalID": signal_id,
                        "cycleNo": cycle_no,
                        "cycleBegin": cycle_begin,
                        "cycleEnd": cycle_end
                    }

                    # Calculate cycle length based on cycle start and end times
                    dict_vehicle_cycle_profile_id["cycleLength"] = abs((dict_vehicle_cycle_profile_id["cycleEnd"] - dict_vehicle_cycle_profile_id["cycleBegin"]).total_seconds())

                    # Process each unique phase number in the current cycle
                    for phase_no in sorted(df_vehicle_phase_profile_cycle["phaseNo"].unique()):
                        # Filter data for the current phase
                        df_vehicle_phase_profile_phase = df_vehicle_phase_profile_cycle[df_vehicle_phase_profile_cycle["phaseNo"] == phase_no]
                        df_vehicle_phase_profile_phase = df_vehicle_phase_profile_phase.reset_index(drop=True)

                        # Initialize phase time columns with NaT for the current phase
                        dict_vehicle_cycle_profile_id.update(
                            {f"{signal_type}Phase{phase_no}": [pd.NaT] for signal_type in ["green", "yellow", "redClearance", "red"]}
                        )

                        # Initialize dictionary to store signal times for each phase
                        dict_signal_types = {
                            signal_type: [] for signal_type in ["green", "yellow", "redClearance", "red"]
                        }

                        # Collect start and end times for green, yellow, and redClearance for each phase 
                        for idx in range(df_vehicle_phase_profile_phase.shape[0]):
                            dict_signal_types["green"].append(
                                tuple([df_vehicle_phase_profile_phase.loc[idx, "greenBegin"], df_vehicle_phase_profile_phase.loc[idx, "greenEnd"]])
                                )
                            dict_signal_types["yellow"].append(
                                tuple([df_vehicle_phase_profile_phase.loc[idx, "yellowBegin"], df_vehicle_phase_profile_phase.loc[idx, "yellowEnd"]])
                                )
                            dict_signal_types["redClearance"].append(
                                tuple([df_vehicle_phase_profile_phase.loc[idx, "redClearanceBegin"], df_vehicle_phase_profile_phase.loc[idx, "redClearanceEnd"]])
                                )

                        # Sort all phase time intervals in order
                        signal_times = [tuple([cycle_begin])] + dict_signal_types["green"] + dict_signal_types["yellow"] + dict_signal_types["redClearance"] + [tuple([cycle_end])]
                        signal_times = sorted(signal_times, key=lambda x: x[0])
                        
                        # Generate 'red' intervals by identifying gaps between sorted intervals
                        for start, end in zip(signal_times[:-1], signal_times[1:]):
                            if start[-1] == end[0]:
                                continue
                            dict_signal_types["red"].append((start[-1], end[0]))

                        # Update cycle information dictionary with collected signal times for each type
                        for signal_type in ["green", "yellow", "redClearance", "red"]:
                            dict_vehicle_cycle_profile_id[f"{signal_type}Phase{phase_no}"] = dict_signal_types[signal_type]

                    # Append the current cycle information to the cycle profile DataFrame
                    df_vehicle_cycle_profile_id = pd.concat([df_vehicle_cycle_profile_id, pd.DataFrame([dict_vehicle_cycle_profile_id])], 
                                                            ignore_index=True)

                # Sort cycle profiles and drop incomplete first and last cycles
                df_vehicle_cycle_profile_id = df_vehicle_cycle_profile_id.sort_values(by=["cycleNo"]).iloc[1:-1].reset_index(drop=True)
                
                # Add date information to the DataFrame
                df_vehicle_cycle_profile_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

            # Path (from database directory) to directory where cycle-level vehicle signal profile will be stored
            _, _, production_signal_dirpath, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_signal",
                signal_id=signal_id
            )

            # Save the phase profile data as a pkl file in the export directory
            export_data(df=df_vehicle_cycle_profile_id, 
                        base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                        sub_dirpath=production_signal_dirpath,
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                        file_type="pkl")

            return df_vehicle_cycle_profile_id

        except Exception as e:
            logging.error(f"Error extracting vehicle cycle profile for signal ID {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting vehicle cycle profile for signal ID {signal_id}: {e}", sys_module=sys)

    def extract_pedestrian_cycle_profile(self, signal_id: str):
        """
        Extracts the pedestrian cycle profile for a specified signal ID and date.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the signal.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the pedestrian cycle profile data.
        """
        try:
            # Path (from database directory) to directory where phase-level pedestrian signal profile is stored
            _, _, production_signal_dirpath, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="phase",
                event_type="pedestrian_signal",
                signal_id=signal_id
            )

            # Load pedestrian phase profile data 
            df_pedestrian_phase_profile_id = load_data(base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                                                       sub_dirpath=production_signal_dirpath, 
                                                       filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                                       file_type="pkl")
            
            # Path (from database directory) to directory where cycle-level vehicle signal profile is stored
            _, _, production_signal_dirpath, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_signal",
                signal_id=signal_id
            )
            
            # Load vehicle cycle profile data 
            df_vehicle_cycle_profile_id = load_data(base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                                                    sub_dirpath=production_signal_dirpath, 
                                                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                                    file_type="pkl")
            
            df_pedestrian_cycle_profile_id = pd.DataFrame()

            if len(df_pedestrian_phase_profile_id) != 0:
                # Keep only rows with a correct pedestrian event sequence (21, 22, 23)
                df_pedestrian_phase_profile_id = df_pedestrian_phase_profile_id[df_pedestrian_phase_profile_id["correctSequenceFlag"] == 1]
                df_pedestrian_phase_profile_id = df_pedestrian_phase_profile_id.reset_index(drop=True)

                # Perform an 'asof' merge to associate each pedestrian walk start with the closest vehicle cycle start
                df_pedestrian_cycle_profile_id = pd.merge_asof(
                    df_pedestrian_phase_profile_id, df_vehicle_cycle_profile_id[["cycleNo", "cycleBegin", "cycleEnd"]], 
                    left_on="pedestrianWalkBegin", 
                    right_on="cycleBegin", 
                    direction="backward"
                )

                # Filter to keep only rows where pedestrianWalkBegin falls within the cycle interval (between cycleBegin and cycleEnd)
                df_pedestrian_cycle_profile_id = df_pedestrian_cycle_profile_id[
                    (df_pedestrian_cycle_profile_id["pedestrianWalkBegin"] >= df_pedestrian_cycle_profile_id["cycleBegin"]) &
                    (df_pedestrian_cycle_profile_id["pedestrianWalkBegin"] <= df_pedestrian_cycle_profile_id["cycleEnd"])
                ]

                # Sort the resulting DataFrame by cycle number and phase number
                df_pedestrian_cycle_profile_id = df_pedestrian_cycle_profile_id.sort_values(by=["cycleNo", "phaseNo"]).reset_index(drop=True)

                # Add date information to the DataFrame
                df_pedestrian_cycle_profile_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

            # Path (from database directory) to directory where cycle-level pedestrian signal profile will be stored
            _, _, production_signal_dirpath, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="pedestrian_signal",
                signal_id=signal_id
            )

            # Save the phase profile data as a pkl file in the export directory
            export_data(df=df_pedestrian_cycle_profile_id, 
                        base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                        sub_dirpath=production_signal_dirpath,
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                        file_type="pkl")

            return df_pedestrian_cycle_profile_id

        except Exception as e:
            # Log and raise an exception if any error occurs during processing
            logging.error(f"Error extracting pedestrian cycle profile for signal ID {signal_id}: {e}")
            raise CustomException(
                custom_message=f"Error extracting pedestrian cycle profile for signal ID {signal_id}: {e}",
                sys_module=sys
            )
        

class SignalFeatureExtract(CoreEventUtils):

    def __init__(self, day: int, month: int, year: int):
        """
        Initialize the class with day, month, and year of the date to process.

        Parameters:
        -----------
        day : int
            Day of the desired date (1-31).
        month : int
            Month of the desired date (1-12).
        year : int
            Year of the desired date (e.g., 2024).
        """
        self.day = day; self.month = month; self.year = year

    def extract_spat(self, signal_id: str):
        """
        Extracts Signal Phasing and Timing (SPaT) data for a given signal ID at the cycle level.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing phase-specific green, yellow, red clearance, and red ratios for each cycle.
        """
        try:
            # Define the path to the directory where vehicle cycle profile data is stored
            _, _, production_signal_dirpath, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_signal",
                signal_id=signal_id
            )

            # Load vehicle cycle profile data
            df_vehicle_cycle_profile_id = load_data(
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_signal_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Initialize variables
            df_spat_id = pd.DataFrame()  # DataFrame to store SPaT data

            # Iterate over each cycle in the vehicle cycle profile
            for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
                # Extract cycle-specific information
                signal_id = df_vehicle_cycle_profile_id.signalID[i]
                cycle_no = df_vehicle_cycle_profile_id.cycleNo[i]
                cycle_begin = df_vehicle_cycle_profile_id.cycleBegin[i]
                cycle_end = df_vehicle_cycle_profile_id.cycleEnd[i]
                cycle_length = df_vehicle_cycle_profile_id.cycleLength[i]

                # Initialize a dictionary to store SPaT data for the current cycle
                dict_spat_id = {
                    "signalID": signal_id,
                    "cycleNo": cycle_no,
                    "cycleBegin": cycle_begin,
                    "cycleEnd": cycle_end,
                    "cycleLength": cycle_length
                }

                # Extract unique phase numbers from the DataFrame columns
                phase_nos = [int(column[-1]) for column in df_vehicle_cycle_profile_id.columns if "Phase" in column]

                # Add placeholders for each signal type duration for each phase
                for phase_no in phase_nos:
                    dict_spat_id.update({
                        f"greenDurationPhase{phase_no}": 0,
                        f"yellowDurationPhase{phase_no}": 0,
                        f"redClearanceDurationPhase{phase_no}": 0,
                        f"redDurationPhase{phase_no}": 0
                    })

                # Calculate signal durations for each phase
                dict_signal_types = {
                    signal_type: [] for signal_type in ["green", "yellow", "redClearance", "red"]
                }  # Temporary storage for signal times

                for phase_no in phase_nos:
                    # Extract red phase times
                    dict_signal_types["red"] = df_vehicle_cycle_profile_id.loc[i, f"redPhase{phase_no}"]
                    if not isinstance(dict_signal_types["red"], list):  # Ensure red times are in list format
                        dict_signal_types["red"] = [dict_signal_types["red"]]

                    # If red times are missing, set default durations
                    if any(pd.isna(time) for time in dict_signal_types["red"]):
                        for signal_type in ["green", "yellow", "redClearance", "red"]:
                            dict_spat_id[f"{signal_type}DurationPhase{phase_no}"] = cycle_length if signal_type == "red" else 0
                        continue

                    # Extract green, yellow, and red clearance times
                    for signal_type in ["green", "yellow", "redClearance"]:
                        dict_signal_types[signal_type] = df_vehicle_cycle_profile_id.loc[i, f"{signal_type}Phase{phase_no}"]
                        if not isinstance(dict_signal_types[signal_type], list):  # Ensure times are in list format
                            dict_signal_types[signal_type] = [dict_signal_types[signal_type]]

                    # Calculate and store signal durations
                    for signal_type in ["green", "yellow", "redClearance", "red"]:
                        time_diff = 0  # Initialize variable to sum time differences
                        for start_time, end_time in dict_signal_types[signal_type]:
                            time_diff += (end_time - start_time).total_seconds()  # Calculate duration in seconds
                        # Calculate and store the duration
                        dict_spat_id[f"{signal_type}DurationPhase{phase_no}"] = round(time_diff, 4)

                # Append the cycle's SPaT data to the DataFrame
                df_spat_id = pd.concat([df_spat_id, pd.DataFrame([dict_spat_id])], axis=0, ignore_index=True)

            # Add date information to the DataFrame
            df_spat_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

            # Cycle-Level
            # Define the output directory for SPaT data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_signal",
                feature_name="spat",
                signal_id=signal_id
            )

            # Save the SPaT data as a pickle file
            export_data(
                df=df_spat_id,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Hourly
            columns = [column for column in df_spat_id.columns if "Duration" in column]
            df_spat_id_hourly = self.calculate_hourly_aggregates(df=df_spat_id, columns=columns)

            # Define the output directory for SPaT data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="hourly",
                event_type="vehicle_signal",
                feature_name="spat",
                signal_id=signal_id
            )

            # Save the SPaT hourly data as a pickle file
            export_data(
                df=df_spat_id_hourly,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Log success and return the data
            logging.info(f"SPaT data successfully extracted and saved for signal ID: {signal_id}")
            return df_spat_id

        except Exception as e:
            # Handle and log any errors
            logging.error(f"Error extracting SPaT data for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting SPaT data for signal ID: {signal_id}", 
                                sys_module=sys)
  

class TrafficFeatureExtract(CoreEventUtils):

    def __init__(self, day: int, month: int, year: int):
        """
        Initialize the class with day, month, and year of the date to process.

        Parameters:
        -----------
        day : int
            Day of the desired date (1-31).
        month : int
            Month of the desired date (1-12).
        year : int
            Year of the desired date (e.g., 2024).
        """
        self.day = day
        self.month = month
        self.year = year 
        logging.info(f"Initialized TrafficFeatureExtract for {year}-{month:02d}-{day:02d}")

    
    def _load_data(self, signal_id: str):
        """
        Load data from various sources required for feature extraction.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the signal.

        Returns:
        --------
        tuple
            DataFrames for event data, configuration data, vehicle cycle profile, and pedestrian cycle profile.
        """
        try:
            logging.info(f"Loading data for signal ID: {signal_id}")

            # Load event data
            interim_event_dirpath, _, _, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(signal_id=signal_id)
            df_event_id = load_data(base_dirpath=os.path.join(root_dir, relative_interim_database_dirpath),
                                    sub_dirpath=interim_event_dirpath, 
                                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                    file_type="pkl")

            # Load configuration data
            _, interim_config_dirpath, _, _ = feature_extraction_dirpath.get_feature_extraction_dirpath()
            df_config_id = load_data(base_dirpath=os.path.join(root_dir, relative_interim_database_dirpath), 
                                     sub_dirpath=interim_config_dirpath,
                                     filename=f"{signal_id}",
                                     file_type="csv")

            # Load vehicle cycle profile data
            _, _, production_signal_dirpath, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_signal",
                signal_id=signal_id
            )
            df_vehicle_cycle_profile_id = load_data(base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                                                    sub_dirpath=production_signal_dirpath, 
                                                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                                    file_type="pkl")
            
            # Load pedestrian cycle profile data
            _, _, production_signal_dirpath, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="pedestrian_signal",
                signal_id=signal_id
            )

            df_pedestrian_cycle_profile_id = load_data(base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                                                       sub_dirpath=production_signal_dirpath, 
                                                       filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                                       file_type="pkl")

            logging.info(f"Successfully loaded data for signal ID: {signal_id}")
            return df_event_id, df_config_id, df_vehicle_cycle_profile_id, df_pedestrian_cycle_profile_id

        except Exception as e:
            logging.error(f"Error loading data for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error loading data for signal ID: {signal_id}", 
                                  sys_module=sys)
    
    def _filter_config_by_detector(self, df_config: pd.DataFrame, detector_type: str):
        """
        Filters the configuration DataFrame based on the detector type (back, front, or count).

        Parameters:
        -----------
        df_config : pd.DataFrame
            The configuration DataFrame containing signal configuration data.
        detector_type : str
            The type of detector to filter ("back", "front", or "count").

        Returns:
        --------
        pd.DataFrame
            The filtered configuration DataFrame.

        Raises:
        -------
        ValueError
            If the detector type is not valid.
        """
        try:
            # Ensure detector type is valid
            if detector_type not in ["back", "front", "count"]:
                raise ValueError(f"Invalid detector type: {detector_type}. Choose 'back', 'front', or 'count'.")

            # Drop rows with missing phase numbers
            df_config = df_config[pd.notna(df_config["phaseNo"])]

            if detector_type == "back":
                # Filter for back detector (stopBarDistance != 0)
                df_config = df_config[df_config["stopBarDistance"] != 0].reset_index(drop=True)

                # If multiple back detectors exist, keep the farthest from the stop bar
                for phase_no in df_config["phaseNo"].unique():
                    df_config_phase = df_config[df_config["phaseNo"] == phase_no]
                    back_detectors_at = df_config_phase["stopBarDistance"].unique().tolist()

                    if len(back_detectors_at) > 1:
                        indices = df_config_phase[df_config_phase["stopBarDistance"] == min(back_detectors_at)].index
                        df_config = df_config.drop(index=indices)

            if detector_type == "front":
                # Filter for front detector (stopBarDistance == 0)
                df_config = df_config[df_config["stopBarDistance"] == 0].reset_index(drop=True)

            if detector_type == "count":
                # Filter for count bar detectors
                df_config = df_config[df_config["countBar"] == 1]
                # df_config = df_config[~df_config["movement"].isin(["R", "TR"])].reset_index(drop=True)

            # Reset index after filtering
            df_config = df_config.reset_index(drop=True)

            return df_config

        except Exception as e:
            logging.error(f"Error filtering configuration data by {detector_type} detector: {e}")
            raise CustomException(custom_message=f"Error filtering configuration data by {detector_type} detector", 
                                sys_module=sys)
        
    def _abbreviate_directions(self, df_config: pd.DataFrame, column_name: str):
        """
        Transforms a column in a DataFrame containing direction strings like
        'Left', 'Left Through', etc., into their abbreviations like 'L', 'LT', etc.
        
        Parameters:
        -----------
        df_config : pd.DataFrame
            The configuration DataFrame containing signal configuration data.
        column_name : str
            The column containing the direction strings (i.e., "laneType").
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with the specified column transformed to abbreviations.
        """
        df_config[column_name] = df_config[column_name].apply(
            lambda value: ''.join(direction[0].upper() for direction in value.split()) if pd.notna(value) else value
        )
        return df_config

    def _calculate_stats(self, df: pd.DataFrame, column_names: list, include_sum: bool = False):
        """
        Processes specified columns in the DataFrame containing nested lists and generates
        derived statistics columns:
        - Minimum value of the nested list values
        - Maximum value of the nested list values
        - Mean value of the nested list values
        - Standard Deviation of the nested list values
        The original columns are retained.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame to process.
        column_names : list
            List of column names in the DataFrame to process.
        include_sum : bool, optional
            If True, revise the columns calculating the sum of each nested list. Default is False.
            If Flase, flatten the values in the columns

        Returns:
        --------
        pd.DataFrame
            DataFrame with additional derived statistics columns and flattened columns with the same name.

        Raises:
        -------
        CustomException
            If an error occurs while processing the columns or column names are not found.
        """
        try:
            # Validate column names
            for col in column_names:
                if col not in df.columns:
                    raise CustomException(
                        custom_message=f"Column '{col}' not found in the DataFrame.",
                        sys_module=sys
                    )
            
            # Initialize dictionary to store derived statistics
            dict_stats = {}

            # Define prefixes for dynamic column name generation
            prefixes = ["green", "yellow", "redClearance", "red"]

            for col in column_names:
                # Identify the prefix in the column name
                prefix = next((p for p in prefixes if col.startswith(p)), None)
                if not prefix:
                    continue  # Skip columns without matching prefixes

                # Generate derived column names with dynamic suffixes
                min_col_name = col.replace(prefix, f"{prefix}Min")
                max_col_name = col.replace(prefix, f"{prefix}Max")
                avg_col_name = col.replace(prefix, f"{prefix}Avg")
                std_col_name = col.replace(prefix, f"{prefix}Std")
                # sum_col_name = col.replace(prefix, f"{prefix}Sum")
                # flt_col_name = col.replace(prefix, f"{prefix}Flt")

                # Calculate the sum of each nested list
                if include_sum:
                    df[col] = df[col].apply(
                        lambda row: [
                            round(np.nansum(lst if not isinstance(lst, float) or not np.isnan(lst) else [0]), 4) 
                            for lst in row
                        ]
                    )
                else:
                    # Flatten the original column in place
                    df[col] = df[col].apply(
                        lambda row: [item for lst in row for item in lst] 
                    )

                # Calculate the minimum of the nested lists
                dict_stats[min_col_name] = df[col].apply(
                    lambda row: 
                        round(np.nanmin(row if not all(np.isnan(row)) else 0), 4) 
                )

                # Calculate the maximum of the sum or flattened lists
                dict_stats[max_col_name] = df[col].apply(
                    lambda row:
                        round(np.nanmax(row if not all(np.isnan(row)) else 0), 4) 
                )

                # Calculate the mean of the sum or flattened list values
                dict_stats[avg_col_name] = df[col].apply(
                    lambda row:
                        round(np.nanmean(row if not all(np.isnan(row)) else 0), 4) 
                )

                # Calculate the standard deviation of the flattened list values
                dict_stats[std_col_name] = df[col].apply(
                    lambda row:
                        round(np.nanstd(row if not all(np.isnan(row)) else 0), 4) 
                )

            # # Drop the original columns
            # df = df.drop(columns=column_names)

            # Use pd.concat to add all derived columns at once
            df = pd.concat([df, pd.DataFrame(dict_stats)], axis=1)

            return df

        except Exception as e:
            raise CustomException(
                custom_message=f"Error processing statistics for columns {column_names}: {str(e)}",
                sys_module=sys
            )

    def _remove_common_periods(self, dict_data: dict):
        """
        Removes common periods between the time intervals of different keys in a dictionary.

        Parameters:
        -----------
        dict_data : dict
            A dictionary where each key contains a list of tuples representing time intervals (start, end).

        Returns:
        --------
        dict
            The modified dictionary where overlapping periods are removed or adjusted to ensure no common periods exist.
        """
        # Helper function to check if two periods overlap
        def is_overlap(period1: tuple, period2: tuple):
            """
            Checks if two time periods overlap.

            Parameters:
            -----------
            period1, period2 : tuple
                Time intervals in the form (start, end).

            Returns:
            --------
            bool
                True if the periods overlap, otherwise False.
            """
            return max(period1[0], period2[0]) < min(period1[1], period2[1])

        # Helper function to adjust overlapping periods
        def adjust_periods(period1: tuple, period2: tuple):
            """
            Adjusts a period to remove overlaps with another period.

            Parameters:
            -----------
            period1 : tuple
                The time period to adjust.
            period2 : tuple
                The time period to compare against.

            Returns:
            --------
            list
                A list of adjusted time intervals with overlaps removed.
            """
            if is_overlap(period1, period2):
                # Case 1: period1 fully contains period2
                if period1[0] < period2[0] and period1[1] > period2[1]:
                    return [(period1[0], period2[0]), (period2[1], period1[1])]
                # Case 2: Overlap at the start
                elif period2[0] <= period1[0] < period2[1]:
                    return [(period2[1], period1[1])]
                # Case 3: Overlap at the end
                elif period2[0] < period1[1] <= period2[1]:
                    return [(period1[0], period2[0])]
            # No overlap case
            return [period1]

        # Extract keys for pairwise comparison
        keys = list(dict_data.keys())

        # Iterate through all pairs of keys
        for i, key1 in enumerate(keys):
            for j, key2 in enumerate(keys):
                # Avoid duplicate or self-comparisons
                if i >= j:
                    continue

                # Initialize a list to store adjusted periods for key1
                new_periods_key1 = []

                # Iterate through periods of key1
                for period1 in dict_data[key1]:
                    # Start with the current period
                    adjusted_periods = [period1]

                    # Compare with periods of key2
                    for period2 in dict_data[key2]:
                        # Adjust the periods to remove overlaps
                        temp = []
                        for p in adjusted_periods:
                            temp.extend(adjust_periods(p, period2))
                        adjusted_periods = temp

                    # Add the adjusted periods to the result
                    new_periods_key1.extend(adjusted_periods)

                # Update the periods for key1
                dict_data[key1] = new_periods_key1

        return dict_data


    def extract_volume(self, signal_id: str, with_countbar: bool = False):
        """
        Extracts cycle-level volume data for a given signal ID.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.
        with_countbar : bool, optional
            If True, uses the count bar filter for configuration data.
            Default is False.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing cycle-level volume data for the specified signal ID.

        Raises:
        -------
        CustomException
            If an error occurs during volume extraction.
        """
        try:
            # Load event, configuration, and vehicle cycle profile data
            df_event_id, df_config_id, df_vehicle_cycle_profile_id, _ = self._load_data(signal_id=signal_id)

            # Filter event data for event sequence [81] (i.e., detector off events)
            df_event_id = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=[81])

            # Drop rows with missing phase numbers in the configuration data and reset the index
            df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])].reset_index(drop=True)
            df_config_id = float_to_int(df_config_id)

            # Apply the appropriate configuration filter based on the with_countbar flag
            if with_countbar:
                df_config_id = self._filter_config_by_detector(df_config=df_config_id, detector_type="count")
            else:
                df_config_id = self._filter_config_by_detector(df_config=df_config_id, detector_type="back")
                df_config_id = self._abbreviate_directions(df_config=df_config_id, column_name="laneType")

            # # Filter configuration for back detector
            # df_config_id = self._filter_config_by_detector(df_config=df_config_id, detector_type="back")

            # Join configuration data with event data on the channel number
            df_event_id = pd.merge(df_event_id, df_config_id,
                                   how="inner",
                                   left_on=["eventParam"], right_on=["channelNo"])

            # Ensure consistent data types for merged data
            df_event_id = float_to_int(df_event_id)

            # Initialize an empty DataFrame to store the final volume data
            df_volume_id = pd.DataFrame()

            # Iterate through each cycle in the vehicle cycle profile data
            for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
                # Extract signal and cycle details
                signal_id = df_vehicle_cycle_profile_id["signalID"][i]
                cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
                cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]
                cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
                cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]

                # Initialize a dictionary to store volume data for the current cycle
                dict_volume_id = {
                    "signalID": signal_id,
                    "cycleNo": cycle_no,
                    "cycleBegin": cycle_begin,
                    "cycleEnd": cycle_end,
                    "cycleLength": cycle_length
                }

                # List all unique phase numbers
                phase_nos = sorted(list(df_event_id["phaseNo"].unique()))
                phase_nos = [
                    phase_no for phase_no in phase_nos if any(str(phase_no) in column for column in df_vehicle_cycle_profile_id.columns)
                ]

                # Initialize phase-specific volume keys in the dictionary
                for phase_no in phase_nos:
                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()

                    for lane_type in lane_types:
                        dict_volume_id.update({
                            f"volumePhase{phase_no}{lane_type}": 0,
                            f"greenVolumePhase{phase_no}{lane_type}": 0,
                            f"yellowVolumePhase{phase_no}{lane_type}": 0,
                            f"redClearanceVolumePhase{phase_no}{lane_type}": 0,
                            f"redVolumePhase{phase_no}{lane_type}": 0
                        })

                # Filter events within the current cycle's time range
                df_event_cycle = df_event_id[
                    (df_event_id["timeStamp"] >= cycle_begin) &
                    (df_event_id["timeStamp"] <= cycle_end)
                ].sort_values(by=["timeStamp", "phaseNo", "channelNo"]).reset_index(drop=True)

                # Process each phase
                for phase_no in phase_nos:
                    # Filter events for the current phase
                    df_event_phase = df_event_cycle[df_event_cycle["phaseNo"] == phase_no].reset_index(drop=True)

                    # Calculate the total number of events for the phase
                    phase_volume = len(df_event_phase)
                    dict_volume_id[f"volumePhase{phase_no}"] = phase_volume

                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()

                    for lane_type in lane_types:
                        channel_nos = df_config_id[((df_config_id["phaseNo"] == phase_no) & 
                                                    (df_config_id["laneType"] == lane_type))]["channelNo"].unique().tolist()

                        # Define signal types
                        signal_types = ["green", "yellow", "redClearance", "red"]

                        # Initialize variable to store volume of the lane type of the given phase
                        volume = 0

                        # Process each signal type
                        for signal_type in signal_types:
                            # Get timestamps for the current signal type and phase
                            timestamps = df_vehicle_cycle_profile_id.loc[i, f"{signal_type}Phase{phase_no}"]

                            # Ensure timestamps are in list format
                            if not isinstance(timestamps, list):
                                timestamps = [timestamps]

                            # Skip processing if no valid timestamps exist
                            if (timestamps == [pd.NaT]) or all(pd.isna(timestamp) for timestamp in timestamps):
                                continue
                            
                            # Initialize variable to store volume of the given signal and lane type of the given phase
                            signal_volume = 0

                            for channel_no in channel_nos:
                                df_event_channel = (
                                    df_event_phase[df_event_phase["channelNo"] == channel_no].reset_index(drop=True)
                                )

                                if len(df_event_channel) == 0:
                                    continue

                                # Calculate the signal-specific volume
                                for start_time, end_time in timestamps:
                                    signal_volume += len(
                                        df_event_channel[
                                            (df_event_channel["timeStamp"] >= start_time) &
                                            (df_event_channel["timeStamp"] <= end_time)
                                        ]
                                    )

                            volume += signal_volume

                            # Store the signal-specific volume in the dictionary
                            dict_volume_id[f"{signal_type}VolumePhase{phase_no}{lane_type}"] = signal_volume
                        
                        # Store the signal-specific volume in the dictionary
                        dict_volume_id[f"volumePhase{phase_no}{lane_type}"] = volume

                # Append the cycle-specific volume data to the DataFrame
                df_volume_id = pd.concat([df_volume_id, pd.DataFrame([dict_volume_id])], axis=0, ignore_index=True)

            # Add date information to the DataFrame
            df_volume_id["date"] = pd.Timestamp(f"{self.year}-{self.month:02d}-{self.day:02d}").date()

            # Cycle-Level
            # Define the directory path to save the volume data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_traffic",
                feature_name="volume",
                signal_id=signal_id
            )

            # Save the volume data as a .pkl file
            export_data(df=df_volume_id,
                        base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                        sub_dirpath=production_feature_dirpath,
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                        file_type="pkl")
            
            # Hourly
            columns = [column for column in df_volume_id.columns if "volume" in column.lower()]
            df_volume_id_hourly = self.calculate_hourly_aggregates(df=df_volume_id, columns=columns, 
                                                                   only_sum=True)

            # Define the directory path to save the volume data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="hourly",
                event_type="vehicle_traffic",
                feature_name="volume",
                signal_id=signal_id
            )

            # Save the volume data as a .pkl file
            export_data(df=df_volume_id_hourly,
                        base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                        sub_dirpath=production_feature_dirpath,
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                        file_type="pkl")

            # Log successful extraction
            logging.info(f"Volume data successfully extracted and saved for signal ID: {signal_id}")
            return df_volume_id

        except Exception as e:
            logging.error(f"Error extracting volume for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting volume for signal ID: {signal_id}", 
                                sys_module=sys)

    def extract_occupancy(self, signal_id: str):
        """
        Extracts occupancy data for a given signal ID at a cycle level.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing occupancy data for each cycle and phase.
        """
        try:
            # Load required data
            df_event_id, df_config_id, df_vehicle_cycle_profile_id, _ = self._load_data(signal_id=signal_id)

            # Filter event data for the event sequence [82, 81] (detector on/off events)
            df_event_id = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=[82, 81])

            # Clean and filter configuration data
            # Remove rows with missing phase numbers
            df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])].reset_index(drop=True)
            df_config_id = float_to_int(df_config_id)

            # Filter configuration data for the stop bar (front detector)
            df_config_id = self._filter_config_by_detector(df_config=df_config_id, detector_type="front")
            df_config_id = self._abbreviate_directions(df_config=df_config_id, column_name="laneType")

            # Join event data with configuration data on channel number
            df_event_id = pd.merge(df_event_id, df_config_id, 
                                   how="inner", 
                                   left_on=["eventParam"], right_on=["channelNo"])

            # Ensure consistent data types
            df_event_id = float_to_int(df_event_id)

            # Initialize an empty DataFrame to store occupancy data
            df_occupancy_id = pd.DataFrame()

            # Process each cycle in the vehicle cycle profile
            for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
                # Extract signal and cycle information
                signal_id = df_vehicle_cycle_profile_id["signalID"][i]
                cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
                cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]
                cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
                cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]

                # Initialize a dictionary to store occupancy data for the current cycle
                dict_occupancy_id = {
                    "signalID": signal_id,
                    "cycleNo": cycle_no,
                    "cycleBegin": cycle_begin,
                    "cycleEnd": cycle_end,
                    "cycleLength": cycle_length
                }

                # List all unique phase numbers
                phase_nos = sorted(list(df_event_id["phaseNo"].unique()))
                phase_nos = [
                    phase_no for phase_no in phase_nos if any(str(phase_no) in column for column in df_vehicle_cycle_profile_id.columns)
                ]

                # Add placeholders for phase-specific occupancy data
                for phase_no in phase_nos:
                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()
                    
                    for lane_type in lane_types:
                        channel_nos = df_config_id[((df_config_id["phaseNo"] == phase_no) & 
                                                    (df_config_id["laneType"] == lane_type))]["channelNo"].unique().tolist()

                        dict_occupancy_id.update({
                            f"channelNos{phase_no}{lane_type}": channel_nos,
                            f"greenOccupancyPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos),
                            f"yellowOccupancyPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos),
                            f"redClearanceOccupancyPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos),
                            f"redOccupancyPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos),
                            f"red5OccupancyPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos)
                        })

                # Set a time tolerance window for capturing overlapping events at cycle boundaries
                if i == 0:  # First cycle
                    st = cycle_begin
                    et = df_vehicle_cycle_profile_id.loc[i+1, "cycleEnd"]
                elif i == len(df_vehicle_cycle_profile_id) - 1:  # Last cycle
                    st = df_vehicle_cycle_profile_id.loc[i-1, "cycleBegin"]
                    et = cycle_end
                else:  # Intermediate cycles
                    st = df_vehicle_cycle_profile_id.loc[i-1, "cycleBegin"]
                    et = df_vehicle_cycle_profile_id.loc[i+1, "cycleEnd"]

                # Filter event data within the tolerance window
                df_event_tolerance = df_event_id[
                    (df_event_id["timeStamp"] >= st) & 
                    (df_event_id["timeStamp"] <= et)
                ].sort_values(by=["timeStamp", "phaseNo", "channelNo"]).reset_index(drop=True)

                # Process each phase and signal type
                for phase_no in phase_nos:
                    # Filter event data for the current phase
                    df_event_phase = df_event_tolerance[df_event_tolerance["phaseNo"] == phase_no].reset_index(drop=True)

                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()

                    # Define signal types to process
                    signal_types = ["green", "yellow", "redClearance", "red", "red5"]

                    for signal_type in signal_types:
                        if signal_type == "red5":
                            # Get timestamps for the current signal type and phase
                            timestamps = df_vehicle_cycle_profile_id.loc[i, f"redClearancePhase{phase_no}"]

                             # Ensure timestamps are in list format
                            if not isinstance(timestamps, list):
                                timestamps = [timestamps]

                            # Skip if no valid timestamps exist
                            if (timestamps == [pd.NaT]) or all(pd.isna(timestamp) for timestamp in timestamps):
                                continue

                            timestamps = [tuple([timestamps[0][0], timestamps[0][0] + pd.Timedelta(seconds=5)])]
                        else:          
                            # Get timestamps for the current signal type and phase
                            timestamps = df_vehicle_cycle_profile_id.loc[i, f"{signal_type}Phase{phase_no}"]

                            # Ensure timestamps are in list format
                            if not isinstance(timestamps, list):
                                timestamps = [timestamps]

                            # Skip if no valid timestamps exist
                            if (timestamps == [pd.NaT]) or all(pd.isna(timestamp) for timestamp in timestamps):
                                continue
                        
                        for lane_type in lane_types:
                            df_config_lane_type = (
                                df_config_id
                                [(df_config_id["phaseNo"] == phase_no) & (df_config_id["laneType"] == lane_type)]
                                .reset_index(drop=True)
                            )

                            lane_nos = df_config_lane_type["laneNoFromLeft"].unique().tolist()

                            indices_to_flatten = []
                            for lane_no in lane_nos:
                                df_config_lane_no = (
                                    df_config_lane_type
                                    [df_config_lane_type["laneNoFromLeft"] == lane_no]
                                    ["channelNo"]
                                    .unique()
                                    .tolist()
                                )
                                channel_nos = df_config_lane_no
                                channel_nos = [channel_no for channel_no in channel_nos if pd.notna(channel_no)]

                                if not channel_nos:
                                    continue
                            
                                dict_occupancy_channel = {
                                    channel_no: [] for channel_no in channel_nos
                                }

                                for channel_no in channel_nos:
                                    df_event_channel = (
                                        df_event_phase[df_event_phase["channelNo"] == channel_no].reset_index(drop=True)
                                    )

                                    if len(df_event_channel) == 0:
                                        continue

                                    # Assign sequence IDs to on/off events
                                    df_event_channel = df_event_channel.copy()
                                    df_event_channel["sequenceID"] = self.add_event_sequence_id(
                                        df_event_channel, valid_event_sequence=[82, 81]
                                    )

                                    for start_time, end_time in timestamps:
                                        for sequence_id in df_event_channel["sequenceID"].unique():
                                            # Filter events for the current sequence
                                            df_event_sequence = (
                                                df_event_channel
                                                [df_event_channel["sequenceID"] == sequence_id]
                                                .reset_index(drop=True)
                                            )

                                            # Skip incomplete sequences (missing on/off events)
                                            if len(df_event_sequence) != 2:
                                                continue

                                            # Extract detector on/off times
                                            detector_ont = df_event_sequence["timeStamp"][0]
                                            detector_oft = df_event_sequence["timeStamp"][1]

                                            if (detector_ont > end_time) or (detector_oft < start_time):
                                                continue

                                            # Calculate overlapping time interval with signal times
                                            max_st = max([detector_ont, start_time])
                                            min_et = min([detector_oft, end_time])

                                            # Calculate occupancy if overlap exists
                                            if max_st < min_et:
                                                dict_occupancy_channel[channel_no].append(
                                                    tuple([max_st, min_et])
                                                )

                                if len(lane_nos) > 1:
                                    dict_occupancy_channel = self._remove_common_periods(
                                        dict_data=dict_occupancy_channel
                                    )

                                for key in dict_occupancy_channel.keys(): 
                                    # Initialize a list to store occupancy data for the current channel
                                    occupancies = [np.nan]

                                    for st_time, en_time in dict_occupancy_channel[key]:
                                        time_diff = abs(round((en_time - st_time).total_seconds(), 4))
                                        occupancies.append(time_diff)

                                    idx = (
                                        dict_occupancy_id[f"channelNos{phase_no}{lane_type}"]
                                        .index(key)
                                    )
                                    
                                    # Store occupancy data for the current channel
                                    dict_occupancy_id[f"{signal_type}OccupancyPhase{phase_no}{lane_type}"][idx] = (
                                        occupancies
                                    ) 
                                
                                if len(channel_nos) > 1:
                                    indices = (
                                        [
                                            dict_occupancy_id[f"channelNos{phase_no}{lane_type}"]
                                            .index(channel_no) for channel_no in channel_nos
                                        ]
                                    )
                                    indices_to_flatten.append(indices)


                            def flatten_dict_value_by_grouped_indices(dict_data, key, grouped_indices):
                                """
                                Flattens specific groups of indices in the list of lists for a given key in a dictionary.

                                Parameters:
                                -----------
                                dict_data : dict
                                    The original dictionary.
                                key : str
                                    The key whose value needs flattening.
                                grouped_indices : list of list of int
                                    Groups of indices specifying which inner lists to flatten together.

                                Returns:
                                --------
                                list
                                    The modified list for the specified key, with the specified groups of indices flattened.
                                """
                                # Ensure the key exists in the dictionary
                                if key not in dict_data:
                                    raise KeyError(f"Key '{key}' not found in the dictionary.")
                                
                                # Extract the value for the key
                                value = dict_data[key]

                                # Keep track of indices to skip once processed
                                flattened_result = []
                                processed_indices = set()

                                for group in grouped_indices:
                                    # Validate indices within the group
                                    valid_indices = [i for i in group if 0 <= i < len(value)]
                                    
                                    # Flatten the group
                                    flattened = []
                                    for i in valid_indices:
                                        if i not in processed_indices:  # Ensure no duplicate processing
                                            flattened += value[i]
                                            processed_indices.add(i)
                                    
                                    # Add the flattened group to the result
                                    if valid_indices:
                                        flattened_result.append(flattened)

                                # Add unprocessed indices back to the result
                                for i in range(len(value)):
                                    if i not in processed_indices:
                                        flattened_result.append(value[i])

                                # Update the dictionary with the modified value
                                dict_data[key] = flattened_result

                                return dict_data[key]
                            
                            if indices_to_flatten:
                                dict_occupancy_id[f"{signal_type}OccupancyPhase{phase_no}{lane_type}"] = (
                                    flatten_dict_value_by_grouped_indices(dict_data=dict_occupancy_id, 
                                                                            key=f"{signal_type}OccupancyPhase{phase_no}{lane_type}", 
                                                                            grouped_indices=indices_to_flatten)
                                )

                # Append the cycle's occupancy data to the DataFrame
                df_occupancy_id = pd.concat([df_occupancy_id, pd.DataFrame([dict_occupancy_id])], axis=0, ignore_index=True)

            # Add date information
            df_occupancy_id["date"] = pd.Timestamp(f"{self.year}-{self.month:02d}-{self.day:02d}").date()

            # Calculate statistics for occupancy data
            columns = [col for col in df_occupancy_id.columns if "Occupancy" in col]
            df_occupancy_id = self._calculate_stats(df=df_occupancy_id, column_names=columns, include_sum=True)

            # Cycle-Level
            # Save the extracted occupancy data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_traffic",
                feature_name="occupancy",
                signal_id=signal_id
            )

            export_data(
                df=df_occupancy_id,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Hourly
            columns = [
                col for col in df_occupancy_id.columns if any(keyword in col for keyword in ["Min", "Max", "Avg", "Std"])
            ]
            df_occupancy_id_hourly = self.calculate_hourly_aggregates(df=df_occupancy_id, columns=columns)

            # Save the extracted occupancy data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="hourly",
                event_type="vehicle_traffic",
                feature_name="occupancy",
                signal_id=signal_id
            )

            export_data(
                df=df_occupancy_id_hourly,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Log success and return the data
            logging.info(f"Occupancy data successfully extracted and saved for signal ID: {signal_id}")
            return df_occupancy_id

        except Exception as e:
            logging.error(f"Error extracting occupancy for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting occupancy for signal ID: {signal_id}", 
                                sys_module=sys)

    def extract_split_failure(self, signal_id: str, purdue_standard: True):
        """
        Extracts split failure information for a given signal ID by comparing SPaT and occupancy data.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.
        purdue_standard: bool
            Determine split failure based on formulation provided by purdue.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing split failure flags for each phase and cycle.
        """
        try:
            # Load SPaT data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_signal",
                feature_name="spat",
                signal_id=signal_id
            )
            df_spat_id = load_data(
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Load occupancy data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_traffic",
                feature_name="occupancy",
                signal_id=signal_id
            )
            df_occupancy_id = load_data(
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Extract green ratio columns from SPaT data
            spat_columns = [col for col in df_spat_id.columns if "greenDurationPhase" in col]

            # Keep only relevant columns in SPaT data
            df_spat_id = df_spat_id[["signalID", "cycleNo", "date", "cycleBegin", "cycleEnd", "cycleLength"] + spat_columns]

            # Extract green occupancy columns from occupancy data
            green_occupancy_columns = [col for col in df_occupancy_id.columns if "greenAvgOccupancyPhase" in col]
            red5_occupancy_columns = [col for col in df_occupancy_id.columns if "redAvg5OccupancyPhase" in col]

            # Keep only relevant columns in occupancy data
            df_occupancy_id = df_occupancy_id[
                ["signalID", "cycleNo", "date", "cycleBegin", "cycleEnd", "cycleLength"] + green_occupancy_columns + red5_occupancy_columns
            ]

            # Merge SPaT and occupancy data on common keys
            df_split_failure_id = pd.merge(
                df_spat_id,
                df_occupancy_id,
                on=["signalID", "cycleNo", "date", "cycleBegin", "cycleEnd", "cycleLength"]
            )

            columns = spat_columns + green_occupancy_columns

            # Extract phase nos
            phase_nos = [col.split('Phase')[-1][0] for col in columns if 'Phase' in col]

            # Find duplicate phases (phases appearing in multiple categories like duration and Occupancy)
            dict_phase_counts = Counter(phase_nos)

            # Keep only those phases that appear at least twice
            phase_nos = [phase_no for phase_no, count in dict_phase_counts.items() if count > 1]

            # # Filter the columns to include only valid phases
            # columns = [col for col in columns if any(f"Phase{phase_no}" in col for phase_no in phase_nos)]
            # columns = sorted(columns, key=lambda x: x[-1])
            
            for phase_no in phase_nos:
                spat_column = next((col for col in spat_columns if phase_no in col))

                for green_occupancy_column in green_occupancy_columns:
                    if phase_no not in green_occupancy_column:
                        continue

                    lane_type = green_occupancy_column.split(f"Phase{phase_no}")[-1]

                    df_split_failure_id[f"greenSplitFailurePhase{phase_no}{lane_type}"] = df_split_failure_id.apply(
                        lambda row: [
                            row[green_occupancy_column] / (row[spat_column] if row[spat_column] != 0 else 1)
                        ],
                        axis=1
                    )

                    if not purdue_standard:
                        # Convert to binary flags: 1 if any occupancy exceeds 100% of green time, else 0
                        df_split_failure_id[f"greenSplitFailurePhase{phase_no}{lane_type}"] = df_split_failure_id[
                            f"greenSplitFailurePhase{phase_no}{lane_type}"
                        ].apply(lambda vals: 1 if any(val >= 1 for val in vals) else 0)
                    else:
                        # Convert to binary flags: 1 if any occupancy exceeds 80% of green time, else 0
                        df_split_failure_id[f"greenSplitFailurePhase{phase_no}{lane_type}"] = df_split_failure_id[
                            f"greenSplitFailurePhase{phase_no}{lane_type}"
                        ].apply(lambda vals: 1 if any(val >= 0.8 for val in vals) else 0)

            if purdue_standard:
                for phase_no in phase_nos:
                    for red5_occupancy_column in red5_occupancy_columns:
                        if f"Phase{phase_no}" not in red5_occupancy_column:
                            continue

                        lane_type = red5_occupancy_column.split(f"Phase{phase_no}")[-1]

                        df_split_failure_id[f"red5SplitFailurePhase{phase_no}{lane_type}"] = df_split_failure_id.apply(
                            lambda row: [
                                row[red5_occupancy_column] / 5
                            ],
                            axis=1
                        )

                        # Convert to binary flags: 1 if any occupancy exceeds 80% of green time, else 0
                        df_split_failure_id[f"red5SplitFailurePhase{phase_no}{lane_type}"] = df_split_failure_id[
                            f"red5SplitFailurePhase{phase_no}{lane_type}"
                        ].apply(lambda vals: 1 if any(val >= 0.8 for val in vals) else 0)

                occupancy_columns = green_occupancy_columns + red5_occupancy_columns 
            else:
                occupancy_columns = green_occupancy_columns
                

            # Drop the intermediate columns for this phase
            df_split_failure_id = df_split_failure_id.drop(columns=spat_columns + occupancy_columns)

            # Cycle-Level
            # Define the path to save split failure data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_traffic",
                feature_name="split_failure",
                signal_id=signal_id
            )

            # Save the split failure data as a pickle file
            export_data(
                df=df_split_failure_id,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Hourly
            columns = [column for column in df_split_failure_id.columns if "SplitFailure" in column]
            df_split_failure_id_hourly = self.calculate_hourly_aggregates(df=df_split_failure_id, columns=columns,
                                                                          only_sum=True)

            # Define the path to save split failure data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="hourly",
                event_type="vehicle_traffic",
                feature_name="split_failure",
                signal_id=signal_id
            )

            # Save the split failure data as a pickle file
            export_data(
                df=df_split_failure_id_hourly,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Return the resulting DataFrame
            return df_split_failure_id

        except Exception as e:
            # Log errors and raise a custom exception for debugging
            logging.error(f"Error extracting split failure data for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting split failure data for signal ID: {signal_id}", 
                                sys_module=sys)

    def extract_headway(self, signal_id: str):
        """
        Extracts headway data for a given signal ID at a cycle level.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing headway data for each cycle and phase.
        """
        try:
            # Load required data
            df_event_id, df_config_id, df_vehicle_cycle_profile_id, _ = self._load_data(signal_id=signal_id)

            # Filter event data for the event code [82] (detector on events)
            df_event_id = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=[82])

            # Clean and filter configuration data
            # Remove rows with missing phase numbers
            df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])].reset_index(drop=True)
            df_config_id = float_to_int(df_config_id)

            # Filter configuration data for the back detector
            df_config_id = self._filter_config_by_detector(df_config=df_config_id, detector_type="back")
            df_config_id = self._abbreviate_directions(df_config=df_config_id, column_name="laneType")

            # Join configuration data with event data on channel number
            df_event_id = pd.merge(df_event_id, df_config_id, 
                                   how="inner", 
                                   left_on=["eventParam"], right_on=["channelNo"])

            # Ensure consistent data types
            df_event_id = float_to_int(df_event_id)

            # Initialize an empty DataFrame to store headway data
            df_headway_id = pd.DataFrame()

            # Process each cycle in the vehicle cycle profile
            for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
                # Extract signal and cycle information
                signal_id = df_vehicle_cycle_profile_id["signalID"][i]
                cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
                cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]
                cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
                cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]

                # Initialize a dictionary to store headway data for the current cycle
                dict_headway_id = {
                    "signalID": signal_id,
                    "cycleNo": cycle_no,
                    "cycleBegin": cycle_begin,
                    "cycleEnd": cycle_end,
                    "cycleLength": cycle_length
                }

                # List all unique phase numbers
                phase_nos = sorted(list(df_event_id["phaseNo"].unique()))
                phase_nos = [
                    phase_no for phase_no in phase_nos if any(str(phase_no) in column for column in df_vehicle_cycle_profile_id.columns)
                ]

                # Add placeholders for phase-specific headway data
                for phase_no in phase_nos:
                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()
                    
                    for lane_type in lane_types:
                        channel_nos = df_config_id[((df_config_id["phaseNo"] == phase_no) & 
                                                    (df_config_id["laneType"] == lane_type))]["channelNo"].unique().tolist()
                        dict_headway_id.update({
                            f"channelNos{phase_no}{lane_type}": channel_nos,
                            f"greenHeadwayPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos),
                            f"yellowHeadwayPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos),
                            f"redClearanceHeadwayPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos),
                            f"redHeadwayPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos)
                        })

                # Filter event data within the current cycle
                df_event_cycle = df_event_id[
                    (df_event_id["timeStamp"] >= cycle_begin) & 
                    (df_event_id["timeStamp"] <= cycle_end)
                ].sort_values(by=["timeStamp", "phaseNo", "channelNo"]).reset_index(drop=True)

                # Process each phase and signal type
                for phase_no in phase_nos:
                    # Filter event data for the current phase
                    df_event_phase = df_event_cycle[df_event_cycle["phaseNo"] == phase_no].reset_index(drop=True)

                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()

                    # Define signal types to process
                    signal_types = ["green", "yellow", "redClearance", "red"]

                    for signal_type in signal_types:
                        # Get timestamps for the current signal type and phase
                        timestamps = df_vehicle_cycle_profile_id.loc[i, f"{signal_type}Phase{phase_no}"]

                        # Ensure timestamps are in list format
                        if not isinstance(timestamps, list):
                            timestamps = [timestamps]

                        # Skip if no valid timestamps exist
                        if (timestamps == [pd.NaT]) or all(pd.isna(timestamp) for timestamp in timestamps):
                            continue

                        for lane_type in lane_types:
                            channel_nos = df_config_id[((df_config_id["phaseNo"] == phase_no) & 
                                                        (df_config_id["laneType"] == lane_type))]["channelNo"].unique().tolist()

                            for channel_no in channel_nos:
                                df_event_channel = (
                                    df_event_phase[df_event_phase["channelNo"] == channel_no].reset_index(drop=True)
                                )

                                if len(df_event_channel) == 0:
                                    continue

                                # Initialize a list to store headway data for the current signal type
                                headways = [np.nan]

                                # Loop through signal times for the current channel
                                for start_time, end_time in timestamps:
                                    # Filter event data within the signal's time range
                                    df_event_signal = df_event_channel[
                                        (df_event_channel["timeStamp"] >= start_time) & 
                                        (df_event_channel["timeStamp"] <= end_time)
                                    ].sort_values(by="timeStamp").reset_index(drop=True)

                                    # Skip if less than two vehicles detected (no headway calculation possible)
                                    if len(df_event_signal) <= 1:
                                        continue

                                    # Calculate headway for consecutive vehicles
                                    for j in range(len(df_event_signal) - 1):
                                        detector_ont_lead = df_event_signal["timeStamp"][j]
                                        detector_ont_next = df_event_signal["timeStamp"][j+1]

                                        # Compute time difference between consecutive vehicles
                                        time_diff = round((detector_ont_next - detector_ont_lead).total_seconds(), 4)
                                        headways.append(time_diff)

                                idx = (
                                    dict_headway_id
                                    [f"channelNos{phase_no}{lane_type}"]
                                    .index(channel_no)
                                )

                                # Store headway data for the current channel
                                dict_headway_id[f"{signal_type}HeadwayPhase{phase_no}{lane_type}"][idx] = headways

                # Append the cycle's headway data to the DataFrame
                df_headway_id = pd.concat([df_headway_id, pd.DataFrame([dict_headway_id])], axis=0, ignore_index=True)

            # Add date information
            df_headway_id["date"] = pd.Timestamp(f"{self.year}-{self.month:02d}-{self.day:02d}").date()

            # Calculate statistics for headway data
            columns = [col for col in df_headway_id.columns if "Headway" in col]
            df_headway_id = self._calculate_stats(df=df_headway_id, column_names=columns, include_sum=False)

            # Cycle-Level
            # Save the extracted headway data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_traffic",
                feature_name="headway",
                signal_id=signal_id
            )
            export_data(
                df=df_headway_id,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Hourly
            columns = [
                col for col in df_headway_id.columns if any(keyword in col for keyword in ["Min", "Max", "Avg", "Std"])
            ]
            df_headway_id_hourly = self.calculate_hourly_aggregates(df=df_headway_id, columns=columns)

            # Save the extracted headway data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="hourly",
                event_type="vehicle_traffic",
                feature_name="headway",
                signal_id=signal_id
            )
            export_data(
                df=df_headway_id_hourly,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Log success and return the data
            logging.info(f"Headway data successfully extracted and saved for signal ID: {signal_id}")
            return df_headway_id

        except Exception as e:
            logging.error(f"Error extracting headway for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting headway for signal ID: {signal_id}", 
                                sys_module=sys)
        
    def extract_conflict(self, signal_id: str, thresholds: list = np.arange(0.5, 5.5, 0.5)):
        """
        Extracts conflict information for a given signal ID using headway data.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.
        thresholds : list, optional
            List of headway thresholds (in seconds) to define conflicts. Default is np.arange(0.5, 5.5, 0.5).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing conflict counts (e.g., counts of headway <0.5, <1.0, etc.) for each phase and cycle.

        Raises:
        -------
        CustomException
            Raised if any error occurs during the extraction process.
        """
        try:
            # Log the start of the conflict extraction process
            logging.info(f"Starting conflict extraction for signal ID: {signal_id}")

            # Load headway data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_traffic",
                feature_name="headway",
                signal_id=signal_id
            )
            df_headway_id = load_data(
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Filter columns containing headway data
            headway_columns = [
                col for col in df_headway_id.columns if "Headway" in col and df_headway_id[col].dtype == "O"
            ]

            # Log the selected columns for conflict calculation
            logging.info(f"Selected headway columns for conflict calculation: {headway_columns}")

            # Define a helper function to calculate conflict counts for each threshold
            def count_conflicts(row, thresholds):
                """
                Counts the number of headway values below each threshold for each phase.

                Parameters:
                -----------
                row : pd.Series
                    Row of DataFrame containing headway data as lists.
                thresholds : list
                    List of thresholds to define conflicts.

                Returns:
                --------
                pd.Series
                    Series containing conflict counts for each threshold.
                """
                conflict_counts = {}
                for column, values in row.items():
                    for threshold in thresholds:
                        count = sum((v < threshold and v > 0) for v in values)
                        conflict_column_name = column.replace("Headway", f"Conflict{threshold}")
                        conflict_counts[conflict_column_name] = count
                return pd.Series(conflict_counts)

            # Apply the conflict calculation function row-wise
            df_conflict_id = df_headway_id[headway_columns].apply(
                count_conflicts, axis=1, thresholds=thresholds
            )

            # Combine conflict counts with metadata columns
            df_conflict_id = pd.concat(
                [df_headway_id[["signalID", "cycleNo", "cycleBegin", "cycleEnd", "cycleLength", "date"]], df_conflict_id],
                axis=1
            )

            # Cycle-Level
            # Define the path to save conflict data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_traffic",
                feature_name="conflict",
                signal_id=signal_id
            )

            # Save the conflict data as a pickle file
            export_data(
                df=df_conflict_id,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Hourly
            columns = [column for column in df_conflict_id.columns if "Conflict" in column]
            df_conflict_id_hourly = self.calculate_hourly_aggregates(df=df_conflict_id, columns=columns, 
                                                                     only_sum=True)

            # Define the path to save conflict data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="hourly",
                event_type="vehicle_traffic",
                feature_name="conflict",
                signal_id=signal_id
            )

            # Save the conflict data as a pickle file
            export_data(
                df=df_conflict_id_hourly,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Return the resulting DataFrame
            return df_conflict_id

        except Exception as e:
            # Log errors and raise a custom exception for debugging
            logging.error(f"Error extracting conflict data for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting conflict data for signal ID: {signal_id}", 
                                sys_module=sys)

    def extract_gap(self, signal_id: str):
        """
        Extracts gap data for a given signal ID at a cycle level.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing gap data for each cycle and phase.
        """
        try:
            # Load required data
            df_event_id, df_config_id, df_vehicle_cycle_profile_id, _ = self._load_data(signal_id=signal_id)

            # Filter event data for the event sequence [81, 82] (detector on/off events)
            df_event_id = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=[81, 82])

            # Clean and filter configuration data
            # Remove rows with missing phase numbers
            df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])].reset_index(drop=True)
            df_config_id = float_to_int(df_config_id)

            # Filter configuration data for the stop bar (front detector)
            df_config_id = self._filter_config_by_detector(df_config=df_config_id, detector_type="back")
            df_config_id = self._abbreviate_directions(df_config=df_config_id, column_name="laneType")

            # Join event data with configuration data on channel number
            df_event_id = pd.merge(df_event_id, df_config_id, 
                                   how="inner", 
                                   left_on=["eventParam"], right_on=["channelNo"])

            # Ensure consistent data types
            df_event_id = float_to_int(df_event_id)

            # Initialize an empty DataFrame to store gaps data
            df_gap_id = pd.DataFrame()

            # Process each cycle in the vehicle cycle profile
            for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
                # Extract signal and cycle information
                signal_id = df_vehicle_cycle_profile_id["signalID"][i]
                cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
                cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]
                cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
                cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]

                # Initialize a dictionary to store occupancy data for the current cycle
                dict_gap_id = {
                    "signalID": signal_id,
                    "cycleNo": cycle_no,
                    "cycleBegin": cycle_begin,
                    "cycleEnd": cycle_end,
                    "cycleLength": cycle_length
                }

                # List all unique phase numbers
                phase_nos = sorted(list(df_event_id["phaseNo"].unique()))
                phase_nos = [
                    phase_no for phase_no in phase_nos if any(str(phase_no) in column for column in df_vehicle_cycle_profile_id.columns)
                ]

                # Add placeholders for phase-specific gap data
                for phase_no in phase_nos:
                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()
                    
                    for lane_type in lane_types:
                        channel_nos = df_config_id[((df_config_id["phaseNo"] == phase_no) & 
                                                    (df_config_id["laneType"] == lane_type))]["channelNo"].unique().tolist()

                        dict_gap_id.update({
                            f"channelNos{phase_no}{lane_type}": channel_nos,
                            f"greenGapPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos),
                            f"yellowGapPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos),
                            f"redClearanceGapPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos),
                            f"redGapPhase{phase_no}{lane_type}": [[np.nan]] * len(channel_nos),
                        })

                # Filter event data within the cycle
                df_event_cycle = df_event_id[
                    (df_event_id["timeStamp"] >= cycle_begin) & 
                    (df_event_id["timeStamp"] <= cycle_end)
                ].sort_values(by=["timeStamp", "phaseNo", "channelNo"]).reset_index(drop=True)

                # Process each phase and signal type
                for phase_no in phase_nos:
                    # Filter event data for the current phase
                    df_event_phase = (
                        df_event_cycle[df_event_cycle["phaseNo"] == phase_no]
                        .reset_index(drop=True)
                    )

                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()

                    # Define signal types to process
                    signal_types = ["green", "yellow", "redClearance", "red"]

                    for signal_type in signal_types: 
                        # Get timestamps for the current signal type and phase
                        timestamps = df_vehicle_cycle_profile_id.loc[i, f"{signal_type}Phase{phase_no}"]

                        # Ensure timestamps are in list format
                        if not isinstance(timestamps, list):
                            timestamps = [timestamps]

                        # Skip if no valid timestamps exist
                        if (timestamps == [pd.NaT]) or all(pd.isna(timestamp) for timestamp in timestamps):
                            continue
                        
                        for lane_type in lane_types:
                            channel_nos = (
                                df_config_id
                                [((df_config_id["phaseNo"] == phase_no) & (df_config_id["laneType"] == lane_type))]
                                ["channelNo"]
                                .unique()
                                .tolist()
                            )

                            for channel_no in channel_nos:
                                df_event_channel = (
                                    df_event_phase[df_event_phase["channelNo"] == channel_no].reset_index(drop=True)
                                )

                                if len(df_event_channel) == 0:
                                    continue

                                gaps = [np.nan]

                                for start_time, end_time in timestamps:
                                    # Filter event data within the signal's time range
                                    df_event_signal = df_event_channel[
                                        (df_event_channel["timeStamp"] >= start_time) & 
                                        (df_event_channel["timeStamp"] <= end_time)
                                    ].sort_values(by="timeStamp").reset_index(drop=True)

                                    # Skip if less than two vehicles detected (no headway calculation possible)
                                    if len(df_event_signal) <= 1:
                                        continue

                                    # Assign sequence IDs to on/off events
                                    df_event_signal = df_event_signal.copy()
                                    df_event_signal["sequenceID"] = self.add_event_sequence_id(
                                        df_event_signal, valid_event_sequence=[81, 82]
                                    )

                                    for sequence_id in df_event_signal["sequenceID"].unique():
                                        # Filter events for the current sequence
                                        df_event_sequence = (
                                            df_event_signal
                                            [df_event_signal["sequenceID"] == sequence_id]
                                            .reset_index(drop=True)
                                        )

                                        # Skip incomplete sequences (missing on/off events)
                                        if len(df_event_sequence) != 2:
                                            continue

                                        # Extract detector off of preceding vehicle & detector on of following vehicle
                                        detector_oft_lead = df_event_sequence["timeStamp"][0]
                                        detector_ont_next = df_event_sequence["timeStamp"][1]

                                        if (detector_oft_lead > end_time) or (detector_ont_next < start_time):
                                            continue

                                        # Calculate time difference (gaps) between consecutive vehicles
                                        time_diff = (
                                            round((detector_ont_next - detector_oft_lead).total_seconds(), 4)
                                        )
                                        gaps.append(time_diff)

                                idx = (
                                    dict_gap_id
                                    [f"channelNos{phase_no}{lane_type}"]
                                    .index(channel_no)
                                )

                                # Store gap data for the current channel
                                dict_gap_id[f"{signal_type}GapPhase{phase_no}{lane_type}"][idx] = gaps

                # Append the cycle's gap data to the DataFrame
                df_gap_id = pd.concat([df_gap_id, pd.DataFrame([dict_gap_id])], axis=0, ignore_index=True)

            # Add date information
            df_gap_id["date"] = pd.Timestamp(f"{self.year}-{self.month:02d}-{self.day:02d}").date()

            # Calculate statistics for gap data
            columns = [col for col in df_gap_id.columns if "Gap" in col]
            df_gap_id = self._calculate_stats(df=df_gap_id, column_names=columns, include_sum=False)

            # Cycle-Level
            # Save the extracted gap data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_traffic",
                feature_name="gap",
                signal_id=signal_id
            )
            export_data(
                df=df_gap_id,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Hourly
            columns = [
                col for col in df_gap_id.columns if any(keyword in col for keyword in ["Min", "Max", "Avg", "Std"])
            ]
            df_gap_id_hourly = self.calculate_hourly_aggregates(df=df_gap_id, columns=columns)

            # Save the extracted gap data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="hourly",
                event_type="vehicle_traffic",
                feature_name="gap",
                signal_id=signal_id
            )
            export_data(
                df=df_gap_id_hourly,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Log success and return the data
            logging.info(f"Headway data successfully extracted and saved for signal ID: {signal_id}")
            return df_gap_id

        except Exception as e:
            logging.error(f"Error extracting gap for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting gap for signal ID: {signal_id}", 
                                sys_module=sys)

    def extract_red_running(self, signal_id: str, with_countbar: bool = False):
        """
        Extracts red light running data for a given signal ID at a cycle level.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.
        with_countbar : bool, optional
            If True, uses the count bar filter for configuration data.
            Default is False.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing red light running data for each cycle and phase.
        """
        try:
            # Load required data
            df_event_id, df_config_id, df_vehicle_cycle_profile_id, _ = self._load_data(signal_id=signal_id)

            # Filter event data for the event sequence 
            if with_countbar:
                # Event sequence [82, 81] (detector on/off events)
                df_event_id = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=[82, 81])
            else:
                # Event code sequence [81] (detector off events)
                df_event_id = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=[81])

            # Clean and filter configuration data
            # Remove rows with missing phase numbers
            df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])].reset_index(drop=True)
            df_config_id = float_to_int(df_config_id)

            # Apply the appropriate configuration filter based on the with_countbar flag
            if with_countbar:
                df_config_id = self._filter_config_by_detector(df_config=df_config_id, detector_type="count")
            else:
                df_config_id = self._filter_config_by_detector(df_config=df_config_id, detector_type="front")
                df_config_id = self._abbreviate_directions(df_config=df_config_id, column_name="laneType")

            # Join configuration data with event data on channel number
            df_event_id = pd.merge(df_event_id, df_config_id, 
                                    how="inner", 
                                    left_on=["eventParam"], right_on=["channelNo"])

            # Ensure consistent data types
            df_event_id = float_to_int(df_event_id)

            # Initialize an empty DataFrame to store red light running data
            df_red_running_id = pd.DataFrame()

            # Process each cycle in the vehicle cycle profile
            for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
                # Extract signal and cycle information
                signal_id = df_vehicle_cycle_profile_id["signalID"][i]
                cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
                cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]
                cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
                cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]

                # Initialize a dictionary to store red light running data for the current cycle
                dict_red_running_id = {
                    "signalID": signal_id,
                    "cycleNo": cycle_no,
                    "cycleBegin": cycle_begin,
                    "cycleEnd": cycle_end,
                    "cycleLength": cycle_length
                }

                # List all unique phase numbers
                phase_nos = sorted(list(df_event_id["phaseNo"].unique()))
                phase_nos = [
                    phase_no for phase_no in phase_nos if any(str(phase_no) in column for column in df_vehicle_cycle_profile_id.columns)
                ]

                # Add placeholders for red light running flags for each phase
                for phase_no in phase_nos:
                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()
                    
                    for lane_type in lane_types:
                        dict_red_running_id.update({
                            f"redClearanceRunningFlagPhase{phase_no}{lane_type}": 0,
                            f"redRunningFlagPhase{phase_no}{lane_type}": 0
                        })

                # Filter event data within the current cycle
                df_event_cycle = df_event_id[
                    (df_event_id["timeStamp"] >= cycle_begin) & 
                    (df_event_id["timeStamp"] <= cycle_end)
                ].sort_values(by=["timeStamp", "phaseNo", "channelNo"]).reset_index(drop=True)

                # Process each phase and signal type
                for phase_no in phase_nos:
                    # Filter event data for the current phase
                    df_event_phase = df_event_cycle[df_event_cycle["phaseNo"] == phase_no].reset_index(drop=True)

                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()

                    # Define signal types to process
                    signal_types = ["redClearance", "red"]

                    for signal_type in signal_types:
                        # Get timestamps for the current signal type and phase
                        timestamps = df_vehicle_cycle_profile_id.loc[i, f"{signal_type}Phase{phase_no}"]

                        # Ensure timestamps are in list format
                        if not isinstance(timestamps, list):
                            timestamps = [timestamps]

                        # Skip if no valid timestamps exist
                        if (timestamps == [pd.NaT]) or all(pd.isna(timestamp) for timestamp in timestamps):
                            continue
                        
                        for lane_type in lane_types:
                            channel_nos = df_config_id[((df_config_id["phaseNo"] == phase_no) & 
                                                        (df_config_id["laneType"] == lane_type))]["channelNo"].unique().tolist()

                            # Initialize a flag to indicate red light running for the current signal type
                            red_running_flag = 0

                            for channel_no in channel_nos:
                                df_event_channel = (
                                    df_event_phase[df_event_phase["channelNo"] == channel_no].reset_index(drop=True)
                                )

                                if len(df_event_channel) == 0:
                                    continue

                                if with_countbar:
                                    # Assign sequence IDs to on/off events
                                    df_event_channel = df_event_channel.copy()
                                    df_event_channel["sequenceID"] = self.add_event_sequence_id(
                                        df_event_channel, valid_event_sequence=[82, 81]
                                    )
                                    
                                    # Loop through signal times for the current channel
                                    for start_time, end_time in timestamps:
                                        for sequence_id in df_event_channel["sequenceID"].unique():
                                            # Filter events for the current sequence
                                            df_event_sequence = df_event_channel[df_event_channel["sequenceID"] == sequence_id].reset_index(drop=True)

                                            # Skip incomplete sequences (missing on/off events)
                                            if len(df_event_sequence) != 2:
                                                continue

                                            # Extract detector on/off times
                                            detector_ont = df_event_sequence["timeStamp"][0]
                                            detector_oft = df_event_sequence["timeStamp"][1]

                                            # Check if the event occurred during the signal's active period
                                            if (detector_ont >= start_time) and (detector_oft <= end_time):
                                                red_running_flag = 1
                                                break
                                else:
                                    # Loop through signal times for the current channel
                                    for start_time, end_time in timestamps:
                                        for j in range(len(df_event_channel)):
                                            # Get the time of the detector off event
                                            timestamp = df_event_channel.loc[j, "timeStamp"]

                                            # Check if the event occurred during the signal's active period
                                            if (timestamp >= start_time) & (timestamp <= end_time):
                                                red_running_flag = 1
                                                break

                                if red_running_flag == 1:
                                    break

                            # Update the flag in the dictionary for the current signal type and phase
                            dict_red_running_id[f"{signal_type}RunningFlagPhase{phase_no}{lane_type}"] = red_running_flag

                # Append the cycle's red light running data to the DataFrame
                df_red_running_id = pd.concat([df_red_running_id, pd.DataFrame([dict_red_running_id])],
                                            axis=0, ignore_index=True)

            # Add date information
            df_red_running_id["date"] = pd.Timestamp(f"{self.year}-{self.month:02d}-{self.day:02d}").date()

            # Cycle-Level
            # Save the extracted red light running data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_traffic",
                feature_name="red_running",
                signal_id=signal_id
            )
            export_data(
                df=df_red_running_id,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Hourly
            columns = [column for column in df_red_running_id.columns if "Flag" in column]
            df_red_running_id_hourly = self.calculate_hourly_aggregates(df=df_red_running_id, columns=columns,
                                                                        only_sum=True)

            # Save the extracted red light running data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="hourly",
                event_type="vehicle_traffic",
                feature_name="red_running",
                signal_id=signal_id
            )
            export_data(
                df=df_red_running_id_hourly,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Log success and return the data
            logging.info(f"Red running data successfully extracted and saved for signal ID: {signal_id}")
            return df_red_running_id

        except Exception as e:
            logging.error(f"Error extracting red running for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting red running for signal ID: {signal_id}", 
                                sys_module=sys)
 
    def extract_dilemma_running(self, signal_id: str, with_countbar: bool = False):
        """
        Extracts dilemma zone running data for a given signal ID at a cycle level.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.
        with_countbar : bool, optional
            If True, uses the count bar filter for configuration data.
            Default is False.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing dilemma zone running data for each cycle and phase.
        """
        try:
            # Load necessary data
            df_event_id, df_config_id, df_vehicle_cycle_profile_id, _ = self._load_data(signal_id=signal_id)

            # Filter event data for the event code sequence [81] (detector off events)
            df_event_id = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=[81])

            # Clean and filter configuration data
            # Remove rows with missing phase numbers
            df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])].reset_index(drop=True)
            df_config_id = float_to_int(df_config_id)

            # Apply the appropriate configuration filter based on the `with_countbar` flag
            if with_countbar:
                df_config_id = self._filter_config_by_detector(df_config=df_config_id, detector_type="count")
            else:
                df_config_id = self._filter_config_by_detector(df_config=df_config_id, detector_type="front")
                df_config_id = self._abbreviate_directions(df_config=df_config_id, column_name="laneType")

            # Merge configuration data with event data based on the channel number
            df_event_id = pd.merge(df_event_id, df_config_id, 
                                   how="inner", 
                                   left_on=["eventParam"], right_on=["channelNo"])

            # Ensure consistent data types after merging
            df_event_id = float_to_int(df_event_id)

            # Initialize an empty DataFrame to store dilemma zone running data
            df_dilemma_running_id = pd.DataFrame()

            # Process each cycle in the vehicle cycle profile
            for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
                # Extract signal and cycle information
                signal_id = df_vehicle_cycle_profile_id["signalID"][i]
                cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
                cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]
                cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
                cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]

                # Initialize a dictionary to store dilemma zone running data for the current cycle
                dict_dilemma_running_id = {
                    "signalID": signal_id,
                    "cycleNo": cycle_no,
                    "cycleBegin": cycle_begin,
                    "cycleEnd": cycle_end,
                    "cycleLength": cycle_length
                }

                # List all unique phase numbers
                phase_nos = sorted(list(df_event_id["phaseNo"].unique()))
                phase_nos = [
                    phase_no for phase_no in phase_nos if any(str(phase_no) in column for column in df_vehicle_cycle_profile_id.columns)
                ]

                # Add placeholders for dilemma zone running flags for each phase
                for phase_no in phase_nos:
                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()
                    
                    for lane_type in lane_types:
                        dict_dilemma_running_id[f"yellowRunningFlagPhase{phase_no}{lane_type}"] = 0

                # Filter event data for the current cycle
                df_event_cycle = df_event_id[
                    (df_event_id["timeStamp"] >= cycle_begin) & 
                    (df_event_id["timeStamp"] <= cycle_end)
                ].sort_values(by=["timeStamp", "phaseNo", "channelNo"]).reset_index(drop=True)

                # Process each phase and signal type
                for phase_no in phase_nos:
                    # Filter event data for the current phase
                    df_event_phase = df_event_cycle[df_event_cycle["phaseNo"] == phase_no].reset_index(drop=True)

                    lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()

                    # Define signal type to process (yellow phase)
                    signal_type = "yellow"

                    # Retrieve timestamps for the yellow phase of the current cycle
                    timestamps = df_vehicle_cycle_profile_id.loc[i, f"{signal_type}Phase{phase_no}"]

                    # Ensure timestamps are in list format
                    if not isinstance(timestamps, list):
                        timestamps = [timestamps]

                    # Skip if no valid timestamps exist
                    if (timestamps == [pd.NaT]) or all(pd.isna(timestamp) for timestamp in timestamps):
                        continue

                    for lane_type in lane_types:
                        channel_nos = df_config_id[((df_config_id["phaseNo"] == phase_no) & 
                                                    (df_config_id["laneType"] == lane_type))]["channelNo"].unique().tolist()

                        # Initialize a flag to indicate dilemma zone running for the current phase
                        dilemma_running_flag = 0

                        for channel_no in channel_nos:
                            df_event_channel = (
                                df_event_phase[df_event_phase["channelNo"] == channel_no].reset_index(drop=True)
                            )

                            if len(df_event_channel) == 0:
                                continue

                            # Loop through yellow phase signal times
                            for start_time, end_time in timestamps:
                                for j in range(len(df_event_channel)):
                                    # Get the timestamp of the detector off event
                                    timestamp = df_event_channel.loc[j, "timeStamp"]

                                    # If the event occurred within the yellow phase signal period, set the flag
                                    if (timestamp >= start_time) & (timestamp <= end_time):
                                        dilemma_running_flag = 1
                                        break
                            
                            if dilemma_running_flag == 1:
                                break
                                

                        # Update the dictionary with the dilemma zone running flag for the current phase
                        dict_dilemma_running_id[f"{signal_type}RunningFlagPhase{phase_no}{lane_type}"] = dilemma_running_flag

                # Append the current cycle's data to the DataFrame
                df_dilemma_running_id = pd.concat([df_dilemma_running_id, pd.DataFrame([dict_dilemma_running_id])],
                                                axis=0, ignore_index=True)

            # Add date information to the DataFrame
            df_dilemma_running_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

            # Cycle-Level
            # Save the extracted dilemma zone running data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="vehicle_traffic",
                feature_name="dilemma_running",
                signal_id=signal_id
            )
            export_data(
                df=df_dilemma_running_id,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Hourly
            columns = [column for column in df_dilemma_running_id.columns if "Flag" in column]
            df_dilemma_running_id_hourly = self.calculate_hourly_aggregates(df=df_dilemma_running_id,
                                                                            columns=columns,
                                                                            only_sum=True)

            # Save the extracted dilemma zone running data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="hourly",
                event_type="vehicle_traffic",
                feature_name="dilemma_running",
                signal_id=signal_id
            )
            export_data(
                df=df_dilemma_running_id_hourly,
                base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                sub_dirpath=production_feature_dirpath,
                filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                file_type="pkl"
            )

            # Log success and return the data
            logging.info(f"Dilemma running data successfully extracted and saved for signal ID: {signal_id}")
            return df_dilemma_running_id

        except Exception as e:
            logging.error(f"Error extracting dilemma running for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting dilemma running for signal ID: {signal_id}", 
                                sys_module=sys)
    
    def extract_pedestrian_activity(self, signal_id: str):
        """
        Extracts cycle-level pedestrian activity data for a given signal ID.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing cycle-level pedestrian activity data for the specified signal ID.

        Raises:
        -------
        CustomException
            If an error occurs during activity extraction.
        """
        try:
            # Load event, configuration, and vehicle cycle profile data
            df_event_id, _, df_vehicle_cycle_profile_id, df_pedestrian_cycle_profile_id = self._load_data(signal_id=signal_id)

            event_codes = [45, 90]
            df_event_id["eventCode"] = df_event_id["eventCode"].astype(int)

            # Filter event data for event sequence [45, 90] (i.e., pedestrian call registered and pedestrian detector off)
            df_event_id = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=event_codes)

            # # Drop duplicates based on time, and event param
            # df_event_id = (
            #     df_event_id.drop_duplicates(subset=["timeStamp", "eventParam"])
            #                .reset_index(drop=True)
            # )

            # Initialize an empty DataFrame to store the final pedestrian activity data
            df_pedestrian_activity_id = pd.DataFrame()

            if len(df_pedestrian_cycle_profile_id) != 0:
                cycle_nos = sorted(df_pedestrian_cycle_profile_id["cycleNo"].unique())  # Get sorted unique cycle numbers
                cycle_nos_vehicle = df_vehicle_cycle_profile_id["cycleNo"].unique()

                # Iterate through each cycle in the vehicle cycle profile data
                for cycle_no in tqdm.tqdm(cycle_nos):
                    df_pedestrian_cycle_profile_cycle = (
                        df_pedestrian_cycle_profile_id[df_pedestrian_cycle_profile_id["cycleNo"] == cycle_no].reset_index(drop=True)
                    )

                    # Extract signal and cycle details
                    signal_id = df_pedestrian_cycle_profile_cycle["signalID"][0]

                    cycle_begin = df_pedestrian_cycle_profile_cycle.loc[0, "cycleBegin"]
                    cycle_end = df_pedestrian_cycle_profile_cycle.loc[0, "cycleEnd"]

                    if (cycle_no - 1) in cycle_nos_vehicle:
                        cycle_begin_prev = (
                            df_vehicle_cycle_profile_id[df_vehicle_cycle_profile_id["cycleNo"] == (cycle_no - 1)]["cycleBegin"].values[0]
                        )
                        cycle_end_prev = (
                            df_vehicle_cycle_profile_id[df_vehicle_cycle_profile_id["cycleNo"] == (cycle_no - 1)]["cycleEnd"].values[0]
                        )

                        cycle_begins = [cycle_begin_prev, cycle_begin]
                        cycle_ends = [cycle_end_prev, cycle_end]
                    else:
                        cycle_begins = [cycle_begin - pd.Timedelta("1.5min"), cycle_begin]
                        cycle_ends = [cycle_begin, cycle_end]                        

                    # Initialize a dictionary to store activity data for the current cycle
                    dict_pedestrian_activity_id = {
                        "signalID": signal_id,
                        "cycleNo": cycle_no,
                        "cycleBegin": cycle_begin,
                        "cycleEnd": cycle_end
                    }

                    # List all unique phase numbers
                    phase_nos = sorted(list(df_event_id["eventParam"].unique()))
                    phase_nos = list(set(phase_nos).intersection(set([2, 4, 8, 6])))

                    # Initialize phase-specific activity keys in the dictionary
                    for phase_no in phase_nos:
                        for event_code in event_codes:
                            dict_pedestrian_activity_id.update({
                                f"pedestrianActivity{event_code}Phase{phase_no}":0,
                                f"pedestrianActivity{event_code}Phase{phase_no}CurrCycle": 0,
                                f"pedestrianActivity{event_code}Phase{phase_no}PrevCycle": 0
                            })

                    cycle_type = ["Prev", "Curr"]

                    for idx, (start_time, end_time) in enumerate(zip(cycle_begins, cycle_ends)):

                        # Filter events within the current cycle's time range
                        df_event_cycle = df_event_id[
                            (df_event_id["timeStamp"] >= start_time) &
                            (df_event_id["timeStamp"] <= end_time)
                        ].sort_values(by=["timeStamp", "eventParam"]).reset_index(drop=True)

                        # Process each phase
                        for phase_no in phase_nos:
                            # Filter events for the current phase
                            df_event_phase = df_event_cycle[df_event_cycle["eventParam"] == phase_no].reset_index(drop=True)

                            for event_code in event_codes:
                                df_event_code = df_event_phase[df_event_phase["eventCode"] == event_code]
                                df_event_code = df_event_code.sort_values(by="timeStamp").reset_index(drop=True)

                                if len(df_event_code) > 1:
                                    base_timestamp = df_event_code.loc[0, "timeStamp"]
                                    indices_to_drop = []
                                    for i in range(1, len(df_event_code)):
                                        timestamp = df_event_code.loc[i, "timeStamp"]

                                        if (timestamp - base_timestamp).total_seconds() < 2.5:
                                            indices_to_drop.append(i)
                                        else:
                                            base_timestamp = timestamp
                                    
                                    df_event_code = df_event_code.drop(index=indices_to_drop) 

                                # Calculate the total number of events for the phase
                                phase_pedestrian_activity = len(df_event_code)
                                dict_pedestrian_activity_id[f"pedestrianActivity{event_code}Phase{phase_no}{cycle_type[idx]}Cycle"] = phase_pedestrian_activity
                                
                                dict_pedestrian_activity_id[f"pedestrianActivity{event_code}Phase{phase_no}"] += phase_pedestrian_activity

                    # Append the cycle-specific pedestrian activity data to the DataFrame
                    df_pedestrian_activity_id = pd.concat([df_pedestrian_activity_id, pd.DataFrame([dict_pedestrian_activity_id])], 
                                                        axis=0, ignore_index=True)

                # Add date information to the DataFrame
                df_pedestrian_activity_id["date"] = pd.Timestamp(f"{self.year}-{self.month:02d}-{self.day:02d}").date()

            # Cycle-Level
            # Define the directory path to save the pedestrian activity data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="pedestrian_traffic",
                feature_name="activity",
                signal_id=signal_id
            )

            # Save the pedestrian activity data as a .pkl file
            export_data(df=df_pedestrian_activity_id,
                        base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                        sub_dirpath=production_feature_dirpath,
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                        file_type="pkl")
            
            # Hourly
            if len(df_pedestrian_cycle_profile_id) != 0:
                columns = [column for column in df_pedestrian_activity_id.columns if "Activity" in column]
                df_pedestrian_activity_id_hourly = self.calculate_hourly_aggregates(df=df_pedestrian_activity_id, 
                                                                                    columns=columns,
                                                                                    only_sum=True)
            else:
                df_pedestrian_activity_id_hourly = pd.DataFrame()

            # Define the directory path to save the pedestrian activity data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="hourly",
                event_type="pedestrian_traffic",
                feature_name="activity",
                signal_id=signal_id
            )

            # Save the pedestrian activity data as a .pkl file
            export_data(df=df_pedestrian_activity_id_hourly,
                        base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                        sub_dirpath=production_feature_dirpath,
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                        file_type="pkl")

            # Log successful extraction
            logging.info(f"Pedestrian activity data successfully extracted and saved for signal ID: {signal_id}")
            return df_pedestrian_activity_id

        except Exception as e:
            logging.error(f"Error extracting pedestrian activity for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting pedestrian activity for signal ID: {signal_id}", 
                                sys_module=sys)
        
    def extract_pedestrian_delay(self, signal_id: str):
        """
        Extracts cycle-level pedestrian delay data for a given signal ID.

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing cycle-level pedestrian delay data for the specified signal ID.

        Raises:
        -------
        CustomException
            If an error occurs during delay extraction.
        """
        try:
            # Load event, configuration, and vehicle cycle profile data
            df_event_id, _, df_vehicle_cycle_profile_id, df_pedestrian_cycle_profile_id = self._load_data(signal_id=signal_id)

            # Filter event data for event sequence [45, 90] (i.e., pedestrian call registered and pedestrian detector off)
            df_event_id = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=[45, 90])

            # Drop duplicates based on time, and event param
            df_event_id = (
                df_event_id.drop_duplicates(subset=["timeStamp", "eventParam"])
                           .reset_index(drop=True)
            )

            # Initialize an empty DataFrame to store the final pedestrian volume data
            df_pedestrian_delay_id = pd.DataFrame()

            if len(df_pedestrian_cycle_profile_id) != 0:
                cycle_nos = sorted(df_pedestrian_cycle_profile_id["cycleNo"].unique())  # Get sorted unique cycle numbers
                cycle_nos_vehicle = df_vehicle_cycle_profile_id["cycleNo"].unique()

                # Iterate through each cycle in the vehicle cycle profile data
                for cycle_no in tqdm.tqdm(cycle_nos):
                    df_pedestrian_cycle_profile_cycle = (
                        df_pedestrian_cycle_profile_id
                        [df_pedestrian_cycle_profile_id["cycleNo"] == cycle_no]
                        .reset_index(drop=True)
                    )

                    # Extract signal and cycle details
                    signal_id = df_pedestrian_cycle_profile_cycle["signalID"][0]

                    cycle_begin = df_pedestrian_cycle_profile_cycle.loc[0, "cycleBegin"]
                    cycle_end = df_pedestrian_cycle_profile_cycle.loc[0, "cycleEnd"]

                    if (cycle_no - 1) in cycle_nos_vehicle:
                        cycle_begin_prev = (
                            df_vehicle_cycle_profile_id[df_vehicle_cycle_profile_id["cycleNo"] == (cycle_no - 1)]["cycleBegin"].values[0]
                        )
                        cycle_end_prev = (
                            df_vehicle_cycle_profile_id[df_vehicle_cycle_profile_id["cycleNo"] == (cycle_no - 1)]["cycleEnd"].values[0]
                        )

                        cycle_begins = [cycle_begin_prev, cycle_begin]
                        cycle_ends = [cycle_end_prev, cycle_end]
                    else:
                        cycle_begins = [cycle_begin - pd.Timedelta("1.5min"), cycle_begin]
                        cycle_ends = [cycle_begin, cycle_end]

                    # Initialize a dictionary to store delay data for the current cycle
                    dict_pedestrian_delay_id = {
                        "signalID": signal_id,
                        "cycleNo": cycle_no,
                        "cycleBegin": cycle_begin,
                        "cycleEnd": cycle_end
                    }

                    # List all unique phase numbers
                    phase_nos = sorted(list(df_event_id["eventParam"].unique()))
                    phase_nos = list(set(phase_nos).intersection(set([2, 4, 8, 6])))

                    # Initialize phase-specific delay keys in the dictionary
                    for phase_no in phase_nos:
                        dict_pedestrian_delay_id.update({
                            f"pedestrianDelayPhase{phase_no}":0,
                            f"pedestrianDelayPhase{phase_no}CurrCycle": 0,
                            f"pedestrianDelayPhase{phase_no}PrevCycle": 0,
                        })
                    
                    # Filter events within the current and previous cycle's time range
                    df_event_tolerance = df_event_id[
                        (df_event_id["timeStamp"] >= cycle_begins[0]) &
                        (df_event_id["timeStamp"] <= cycle_ends[-1])
                    ].sort_values(by=["timeStamp", "eventParam"]).reset_index(drop=True)

                    cycle_types = ["Prev", "Curr"]

                    for idx, (start_time, end_time) in enumerate(zip(cycle_begins, cycle_ends)):
                        # Process each phase
                        for phase_no in phase_nos:
                            # Filter events for the current phase
                            df_event_phase = df_event_tolerance[df_event_tolerance["eventParam"] == phase_no].reset_index(drop=True)

                            button_press_times = df_event_phase["timeStamp"].unique().tolist()

                            if not button_press_times:  # Check if the list is empty
                                continue

                            button_press_time = None

                            if (cycle_no - 1) in cycle_nos:
                                df_pedestrian_cycle_profile_phase_prev = (
                                    df_pedestrian_cycle_profile_id[
                                        (df_pedestrian_cycle_profile_id["cycleNo"] == (cycle_no - 1)) & 
                                        (df_pedestrian_cycle_profile_id["phaseNo"] == phase_no)
                                    ].reset_index(drop=True)
                                )

                                clearance_ends_prev = (
                                    df_pedestrian_cycle_profile_phase_prev
                                    ["pedestrianClearanceEnd"]
                                    .unique()
                                )
                                
                                if clearance_ends_prev:
                                    clearance_end_prev = max(clearance_ends_prev)

                                    # Iterate through button_press_times to find the next valid time
                                    for time in button_press_times:
                                        if time > clearance_end_prev:  # Check if the button press time is greater
                                            button_press_time = time
                                            break
                            
                            if button_press_time is None:
                                button_press_time = min(button_press_times)                            

                            df_pedestrian_cycle_profile_phase = (
                                df_pedestrian_cycle_profile_cycle
                                [df_pedestrian_cycle_profile_cycle["phaseNo"] == phase_no]
                                .reset_index(drop=True)
                            )

                            for i in range(len(df_pedestrian_cycle_profile_phase)):
                                walk_begin = df_pedestrian_cycle_profile_phase.loc[i, "pedestrianWalkBegin"]

                                st = max(button_press_time, start_time)
                                et = min(walk_begin, end_time)

                                if et > st:
                                    phase_pedestrian_delay = round((et - st).total_seconds(), 4)
                                else:
                                    continue

                                # Calculate the delay for the phase
                                dict_pedestrian_delay_id[f"pedestrianDelayPhase{phase_no}{cycle_types[idx]}Cycle"] = phase_pedestrian_delay
                                
                                dict_pedestrian_delay_id[f"pedestrianDelayPhase{phase_no}"] += phase_pedestrian_delay


                    # Append the cycle-specific pedestrian delay data to the DataFrame
                    df_pedestrian_delay_id = pd.concat([df_pedestrian_delay_id, pd.DataFrame([dict_pedestrian_delay_id])], 
                                                        axis=0, ignore_index=True)

                # Add date information to the DataFrame
                df_pedestrian_delay_id["date"] = pd.Timestamp(f"{self.year}-{self.month:02d}-{self.day:02d}").date()

            # Cycle-Level
            # Define the directory path to save the pedestrian delay data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="pedestrian_traffic",
                feature_name="delay",
                signal_id=signal_id
            )

            # Save the pedestrian delay data as a .pkl file
            export_data(df=df_pedestrian_delay_id,
                        base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                        sub_dirpath=production_feature_dirpath,
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                        file_type="pkl")
            
            # Hourly
            if len(df_pedestrian_cycle_profile_id) != 0:
                columns = [column for column in df_pedestrian_delay_id.columns if "Delay" in column]
                df_pedestrian_delay_id_hourly = self.calculate_hourly_aggregates(df=df_pedestrian_delay_id, 
                                                                                 columns=columns)
            else:
                df_pedestrian_delay_id_hourly = pd.DataFrame()

            # Define the directory path to save the pedestrian delay data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="hourly",
                event_type="pedestrian_traffic",
                feature_name="delay",
                signal_id=signal_id
            )

            # Save the pedestrian delay data as a .pkl file
            export_data(df=df_pedestrian_delay_id_hourly,
                        base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                        sub_dirpath=production_feature_dirpath,
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                        file_type="pkl")

            # Log successful extraction
            logging.info(f"Pedestrian delay data successfully extracted and saved for signal ID: {signal_id}")
            return df_pedestrian_delay_id

        except Exception as e:
            logging.error(f"Error extracting pedestrian delay for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting pedestrian delay for signal ID: {signal_id}", 
                                sys_module=sys)
        
    def extract_turn_conflict_intensity(self, signal_id: str):
        """
        Extracts cycle-level vehicle-pedestrian conflict intensity for a given signal ID. Specifically, 
        extracts conflict insights of righ-turn vehicle with pedestrian activity.
        The insights can be extracted using various methods considering the type of detector configuration for 
        right-turn lanes (i.e., 'Right', 'Through Right', 'Left Right', or 'Left Through Right'). 

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the traffic signal.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing cycle-level vehicle-pedestrian conflict intensity for the specified signal ID.

        Raises:
        -------
        CustomException
            If an error occurs during vehicle-pedestrian conflict intensity extraction.
        """
        try:
            # Load event, configuration, and vehicle cycle profile data
            df_event_id, df_config_id, df_vehicle_cycle_profile_id, df_pedestrian_cycle_profile_id = self._load_data(signal_id=signal_id)

            dict_map = {
                1: 6, 5: 2, 3: 8, 7: 4
            }

            df_config_id = float_to_int(df_config_id)
            df_config_id = self._abbreviate_directions(df_config=df_config_id, column_name="laneType")
            df_config_id_back = self._filter_config_by_detector(df_config=df_config_id, detector_type="back")

            indices_to_drop = []
            for approach in df_config_id["approach"].unique().tolist():
                df_config_approach = df_config_id[df_config_id["approach"] == approach]

                phase_nos = df_config_approach["phaseNo"].unique().tolist()

                for phase_no in phase_nos:
                    if phase_no % 2 == 0:
                        break
                    else: 
                        phase_no = dict_map[phase_no]
                        break

                for idx in df_config_approach.index.values:
                    if "R" not in df_config_approach.loc[idx, "laneType"]:
                        indices_to_drop.append(idx)

                    df_config_id = df_config_id.copy()
                    df_config_id.loc[idx:idx+1, "phaseNo"] = df_config_id.loc[idx:idx+1, "phaseNo"].fillna(phase_no)

            # Filter event data for event sequence [45, 90] (i.e., pedestrian call registered and pedestrian detector off)
            df_event_id_pedestrian = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=[45, 90])

            # Drop duplicates based on time, and event param
            df_event_id_pedestrian = (
                df_event_id_pedestrian.drop_duplicates(subset=["timeStamp", "eventParam"])
                .reset_index(drop=True)
            )

            df_config_id = df_config_id.drop(index=indices_to_drop)

            # Initialize an empty DataFrame to store the final turn conflict intensity data
            df_turn_conflict_intensity_id = pd.DataFrame()

            if len(df_pedestrian_cycle_profile_id) != 0:
                cycle_nos = sorted(df_pedestrian_cycle_profile_id["cycleNo"].unique())  # Get sorted unique cycle numbers
                cycle_nos_vehicle = df_vehicle_cycle_profile_id["cycleNo"].unique()

                # Iterate through each cycle in the vehicle cycle profile data
                for cycle_no in tqdm.tqdm(cycle_nos):
                    df_pedestrian_cycle_profile_cycle = (
                        df_pedestrian_cycle_profile_id[df_pedestrian_cycle_profile_id["cycleNo"] == cycle_no].reset_index(drop=True)
                    )

                    # Extract signal and cycle details
                    signal_id = df_pedestrian_cycle_profile_cycle["signalID"][0]

                    cycle_begin = df_pedestrian_cycle_profile_cycle.loc[0, "cycleBegin"]
                    cycle_end = df_pedestrian_cycle_profile_cycle.loc[0, "cycleEnd"]

                    if (cycle_no - 1) in cycle_nos_vehicle:
                        cycle_begin_prev = (
                            df_vehicle_cycle_profile_id[df_vehicle_cycle_profile_id["cycleNo"] == (cycle_no - 1)]["cycleBegin"].values[0]
                        )
                        cycle_end_prev = (
                            df_vehicle_cycle_profile_id[df_vehicle_cycle_profile_id["cycleNo"] == (cycle_no - 1)]["cycleEnd"].values[0]
                        )
                        
                        cycle_begins = [cycle_begin_prev, cycle_begin]
                        cycle_ends = [cycle_end_prev, cycle_end]
                    else:
                        cycle_begins = [cycle_begin - pd.Timedelta("1.5min"), cycle_begin]
                        cycle_ends = [cycle_begin, cycle_end]

                    # Initialize a dictionary to store turn conflict intensity data for the current cycle
                    dict_turn_conflict_intensity_id = {
                        "signalID": signal_id,
                        "cycleNo": cycle_no,
                        "cycleBegin": cycle_begin,
                        "cycleEnd": cycle_end,
                    }

                    # Filter events within the current and previous cycle's time range
                    df_event_tolerance_pedestrian = (
                        df_event_id_pedestrian[(df_event_id_pedestrian["timeStamp"] >= cycle_begins[0]) & 
                                               (df_event_id_pedestrian["timeStamp"] <= cycle_ends[-1])
                                   ]
                        .sort_values(by=["timeStamp", "eventParam"])
                        .reset_index(drop=True)
                    )

                    phase_nos = df_config_id["phaseNo"].unique().tolist()

                    # Initialize phase-specific turn conflict intensity keys in the dictionary
                    for phase_no in phase_nos:
                        dict_turn_conflict_intensity_id.update(
                            {
                                f"pedestrianActivityBeginPhase{phase_no}": pd.NaT,
                                f"pedestrianActivityEndPhase{phase_no}": pd.NaT,
                                f"pedestrianActivityPhase{phase_no}":0,
                                f"pedestrianActivityDurationPhase{phase_no}":0,
                            }
                        )
                        lane_types = df_config_id[df_config_id["phaseNo"] == phase_no]["laneType"].unique().tolist()
                        
                        for lane_type in lane_types:
                            dict_turn_conflict_intensity_id.update({
                                f"vehicleOccupancyPhase{phase_no}{lane_type}":0,
                                f"vehicleVolumePhase{phase_no}{lane_type}":0,
                            })

                    for phase_no in phase_nos:
                        df_config_phase = df_config_id[df_config_id["phaseNo"] == phase_no]
                        df_config_phase = df_config_phase.reset_index(drop=True)
                        
                        df_pedestrian_cycle_profile_phase = (
                            df_pedestrian_cycle_profile_cycle
                            [
                                df_pedestrian_cycle_profile_cycle["phaseNo"] == phase_no
                            ]
                        )

                        if (len(df_config_phase) == 0) or (len(df_pedestrian_cycle_profile_phase) == 0):
                            continue

                        # Get pedestrian activity duration
                        activity_end_time = (
                            df_pedestrian_cycle_profile_phase.loc[df_pedestrian_cycle_profile_phase.index[-1], "pedestrianDontWalkBegin"]
                        )

                        # Filter events for the current phase
                        df_event_phase_pedestrian = (
                            df_event_tolerance_pedestrian[df_event_tolerance_pedestrian["eventParam"] == phase_no]
                            .reset_index(drop=True)
                        )

                        if len(df_event_phase_pedestrian) > 1:
                            base_timestamp = df_event_phase_pedestrian.loc[0, "timeStamp"]
                            indices_to_drop = []

                            for i in range(1, len(df_event_phase_pedestrian)):
                                timestamp = df_event_phase_pedestrian.loc[i, "timeStamp"]

                                if (timestamp - base_timestamp).total_seconds() < 2.5:
                                    indices_to_drop.append(i)
                                else:
                                    base_timestamp = timestamp
                            
                            df_event_phase_pedestrian = df_event_phase_pedestrian.drop(index=indices_to_drop) 
                            df_event_phase_pedestrian = df_event_phase_pedestrian.reset_index(drop=True)
                        
                        dict_turn_conflict_intensity_id[f"pedestrianActivityPhase{phase_no}"] = (
                            len(df_event_phase_pedestrian)
                        )

                        for start_time, end_time in zip(cycle_begins, cycle_ends):
                            df_event_cycle_pedestrian = (
                                df_event_phase_pedestrian
                                [
                                    (df_event_phase_pedestrian["timeStamp"] >= start_time) &
                                    (df_event_phase_pedestrian["timeStamp"] <= end_time)
                                ]
                                .reset_index(drop=True)
                            )

                            if len(df_event_cycle_pedestrian) == 0:
                                activity_start_time = end_time
                                break
                            else:
                                df_pedestrian_cycle_profile_phase_prev = (
                                    df_pedestrian_cycle_profile_id
                                    [
                                        (df_pedestrian_cycle_profile_id["cycleNo"] == (cycle_no - 1)) & 
                                        (df_pedestrian_cycle_profile_id["phaseNo"] == phase_no)
                                    ]
                                    .reset_index(drop=True)
                                )

                                if len(df_pedestrian_cycle_profile_phase_prev) != 0:
                                    clearance_ends_prev = (
                                        df_pedestrian_cycle_profile_phase_prev
                                        ["pedestrianClearanceEnd"]
                                        .unique()
                                    )
                                    
                                    if clearance_ends_prev:
                                        clearance_end_prev = max(clearance_ends_prev)
                                        for i in range(len(df_event_cycle_pedestrian)):
                                            button_press_time = df_event_cycle_pedestrian.loc[i, "timeStamp"]
                                            if clearance_end_prev < button_press_time:
                                                activity_start_time = button_press_time 
                                                break 
                                else:
                                    button_press_times = df_event_cycle_pedestrian["timeStamp"].unique().tolist()
                                    activity_start_time = min(button_press_times) 
                                    break 
                        
                        activity_start_time = (
                            activity_start_time - pd.Timedelta("0.5min")
                        )
                        activity_end_time = (
                            activity_end_time + pd.Timedelta("0.5min")
                        )

                        dict_turn_conflict_intensity_id[f"pedestrianActivityBeginPhase{phase_no}"] = (
                            activity_start_time
                        )
                        dict_turn_conflict_intensity_id[f"pedestrianActivityEndPhase{phase_no}"] = (
                            activity_end_time
                        )

                        df_config_phase_front = (
                            self._filter_config_by_detector(df_config=df_config_phase, detector_type="front")
                        )

                        dict_turn_conflict_intensity_id[f"pedestrianActivityDurationPhase{phase_no}"] = (
                            round((activity_end_time - activity_start_time).total_seconds(), 4)
                        )

                        # Vehicle Occupancy
                        if (len(df_config_phase_front) != 0):
                            lane_types = (
                                df_config_phase_front["laneType"].unique().tolist()
                            )

                            for lane_type in lane_types:
                                df_config_lane_type = (
                                    df_config_phase_front
                                    [
                                        df_config_phase_front["laneType"] == lane_type
                                    ]
                                    .reset_index(drop=True)
                                )

                                lane_nos = df_config_lane_type["laneNoFromLeft"].unique().tolist()

                                vehicle_occupancies = [np.nan]
                                for lane_no in lane_nos:
                                    df_config_lane_no = (
                                        df_config_lane_type
                                        [
                                            df_config_lane_type["laneNoFromLeft"] == lane_no
                                        ]
                                        .reset_index(drop=True)
                                    )
                                    channel_nos = df_config_lane_no["channelNo"].unique().tolist()
                                    channel_nos = [channel_no for channel_no in channel_nos if pd.notna(channel_no)]

                                    if not channel_nos:
                                        continue

                                    vehicle_occupancy = 0 
                                    for channel_no in channel_nos:
                                        df_config_channel = (
                                            df_config_lane_no
                                            [
                                                df_config_lane_no["channelNo"] == channel_no
                                            ]
                                            .reset_index(drop=True)
                                        )

                                        if len(df_config_channel) == 0:
                                            continue

                                        df_event_channel_vehicle = (
                                            df_event_id                     
                                            [
                                                (df_event_id["timeStamp"] >= activity_start_time) & 
                                                (df_event_id["timeStamp"] <= activity_end_time )
                                            ]
                                            .copy()
                                            .pipe(self.filter_by_event_sequence, event_sequence=[82, 81])
                                            .sort_values(by=["timeStamp", "eventParam"])
                                            .reset_index(drop=True)
                                        )

                                        if len(df_event_channel_vehicle) == 0:
                                            continue

                                        df_event_channel_vehicle = df_event_channel_vehicle.copy()

                                        # Join event data with configuration data on channel number
                                        df_event_channel_vehicle = (
                                            df_event_channel_vehicle
                                            .merge(
                                                df_config_channel,
                                                how="inner",
                                                left_on="eventParam",
                                                right_on="channelNo"
                                            )
                                            .pipe(float_to_int)  # Ensure consistent data types
                                        )

                                        # Assign sequence IDs to on/off events
                                        df_event_channel_vehicle = df_event_channel_vehicle.copy()
                                        df_event_channel_vehicle["sequenceID"] = self.add_event_sequence_id(
                                            df_event_channel_vehicle, valid_event_sequence=[82, 81]
                                        )

                                        for sequence_id in df_event_channel_vehicle["sequenceID"].unique():
                                        # Filter events for the current sequence
                                            df_event_sequence_vehicle = (
                                                df_event_channel_vehicle
                                                [
                                                    df_event_channel_vehicle["sequenceID"] == sequence_id
                                                ]
                                                .reset_index(drop=True)
                                            )

                                            # Skip incomplete sequences (missing on/off events)
                                            if len(df_event_sequence_vehicle) != 2:
                                                continue

                                            # Extract detector on/off times
                                            detector_ont = df_event_sequence_vehicle["timeStamp"][0]
                                            detector_oft = df_event_sequence_vehicle["timeStamp"][1]

                                            # Calculate occupancy 
                                            if detector_ont < detector_oft:
                                                time_diff = round((detector_oft - detector_ont).total_seconds(), 4)
                                                vehicle_occupancy += time_diff

                                    vehicle_occupancies.append(vehicle_occupancy)
                                
                                dict_turn_conflict_intensity_id[f"vehicleOccupancyPhase{phase_no}{lane_type}"] = (
                                    round(np.nanmean(vehicle_occupancies), 4)
                                )

                        df_config_phase_back = (
                            self._filter_config_by_detector(df_config=df_config_phase, detector_type="back")
                        )

                        # Vehicle Volume
                        if (len(df_config_phase_back) != 0):
                            lane_types = (
                                df_config_phase_back["laneType"].unique().tolist()
                            )

                            for lane_type in lane_types:
                                channel_nos = (
                                    df_config_phase_back
                                    [
                                        (df_config_phase_back["laneType"] == lane_type)
                                    ]
                                    ["channelNo"].unique().tolist()
                                )
                                
                                channel_nos = [channel_no for channel_no in channel_nos if pd.notna(channel_no)]

                                if not channel_nos:
                                    continue

                                vehicle_volume = 0 
                                for channel_no in channel_nos:
                                    df_config_channel_back = (
                                        df_config_phase_back[df_config_phase_back["channelNo"] == channel_no]
                                        .reset_index(drop=True)
                                    )

                                    df_event_channel_vehicle = (
                                        df_event_id                     
                                        [
                                            (df_event_id["timeStamp"] >= activity_start_time) & 
                                            (df_event_id["timeStamp"] <= activity_end_time )
                                        ]
                                        .copy()
                                        .pipe(self.filter_by_event_sequence, event_sequence=[81])
                                        .sort_values(by=["timeStamp", "eventParam"])
                                        .reset_index(drop=True)
                                    )

                                    if len(df_event_channel_vehicle) == 0:
                                        continue

                                    df_event_channel_vehicle = df_event_channel_vehicle.copy()

                                    # Join event data with configuration data on channel number
                                    df_event_channel_vehicle = (
                                        df_event_channel_vehicle
                                        .merge(
                                            df_config_channel_back,
                                            how="inner",
                                            left_on="eventParam",
                                            right_on="channelNo"
                                        )
                                        .pipe(float_to_int)  # Ensure consistent data types
                                    )

                                    if len(df_event_channel_vehicle) == 0:
                                        continue

                                    vehicle_volume += len(df_event_channel_vehicle)
                            
                            dict_turn_conflict_intensity_id[f"vehicleVolumePhase{phase_no}{lane_type}"] = (
                                vehicle_volume
                            )       

                    # Append the cycle-specific turn conflict intensity data to the DataFrame
                    df_turn_conflict_intensity_id = (
                        pd.concat([df_turn_conflict_intensity_id, pd.DataFrame([dict_turn_conflict_intensity_id])], 
                                   axis=0, ignore_index=True)
                    )

                # Add date information to the DataFrame
                df_turn_conflict_intensity_id["date"] = (
                    pd.Timestamp(f"{self.year}-{self.month:02d}-{self.day:02d}").date()
                )

            # Cycle-Level
            # Define the directory path to save the turn conflict intensity data
            _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
                resolution_level="cycle",
                event_type="pedestrian_traffic",
                feature_name="turn_conflict_intensity",
                signal_id=signal_id
            )

            # Save the turn conflict intensity data as a .pkl file
            export_data(df=df_turn_conflict_intensity_id,
                        base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                        sub_dirpath=production_feature_dirpath,
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}",
                        file_type="pkl")
            
            # Log successful extraction
            logging.info(f"Turn conflict intensity data successfully extracted and saved for signal ID: {signal_id}")
            return df_turn_conflict_intensity_id

        except Exception as e:
            logging.error(f"Error extracting turn conflict intensity for signal ID: {signal_id}: {e}")
            raise CustomException(custom_message=f"Error extracting turn conflict intensity for signal ID: {signal_id}", 
                                sys_module=sys)









