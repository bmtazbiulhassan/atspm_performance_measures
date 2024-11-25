import pandas as pd
import numpy as np
import yaml
import tqdm
import sys
import os

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
                # Filter data for each phase number and assign sequence IDs
                df_event_phase = df_event_id[df_event_id[dict_column_names["param"]] == phase_no]
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

            # Assign cycle numbers to the phase profile based on the specified start barrier
            df_vehicle_phase_profile_id = self.assign_cycle_nos(df_vehicle_phase_profile=df_vehicle_phase_profile_id, start_barrier_no=start_barrier_no)
            cycle_nos = sorted(df_vehicle_phase_profile_id["cycleNo"].unique())  # Get sorted unique cycle numbers

            df_vehicle_cycle_profile_id = pd.DataFrame()  # Initialize DataFrame for cycle profiles

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

    def extract_spat(self, signal_id: str): # Signal Phasing and Timing (SPAT)
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

        # List of signals
        signal_types = ["green", "yellow", "redClearance", "red"]
        
        # Initialize dataframe to store phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio per cycle
        df_spat_id = pd.DataFrame() # stratio: signal (i.e., "green", "yellow", "redClearance", and "red") ratio
        
        for i in range(len(df_vehicle_cycle_profile_id)):
            # Extract signal info: cycle
            signal_id = df_vehicle_cycle_profile_id.signalID[i]
            
            cycle_no = df_vehicle_cycle_profile_id.cycleNo[i]
            cycle_begin = df_vehicle_cycle_profile_id.cycleBegin[i]; cycle_end = df_vehicle_cycle_profile_id.cycleEnd[i]
            cycle_length = df_vehicle_cycle_profile_id.cycleLength[i]
        
            # Initialize a dictionary with signal info to also store phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio per cycle
            dict_spat_id = {
                "signalID": "0", "cycleNo": 0, "cycleBegin": pd.NaT, "cycleEnd": pd.NaT, "cycleLength": 0
                } 
        
            # Get list of unique phase nos from columns
            phase_nos = [int(column[-1]) for column in df_vehicle_cycle_profile_id.columns if "Phase" in column]
        
            # Update dictionary to hold phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio per cycle
            for phase_no in phase_nos:
                dict_spat_id.update(
                    {f"greenRatioPhase{phase_no}": 0, 
                     f"yellowRatioPhase{phase_no}": 0, 
                     f"redClearanceRatioPhase{phase_no}": 0, f"redRatioPhase{phase_no}": 0}
                )
        
            # Add signal info to the dictonary
            dict_spat_id["signalID"] = signal_id; 
            
            dict_spat_id["cycleNo"] = cycle_no
            dict_spat_id["cycleBegin"] = cycle_begin; dict_spat_id["cycleEnd"] = cycle_end
            dict_spat_id["cycleLength"] = cycle_length
        
            # Intialize dictionary to temporarily add signal times of every phase
            dict_signal_types = {"green": [], "yellow": [], "redClearance": [], "red": []}
            
            # Loop through phase nos 
            for phase_no in phase_nos:
                # Get red times
                dict_signal_types["red"] = df_vehicle_cycle_profile_id.loc[i, f"redPhase{phase_no}"]

                # Check if the instance is not list (if not, convert to list)
                if not isinstance(dict_signal_types["red"], list):
                    dict_signal_types["red"] = [dict_signal_types["red"]]
        
                # If there's no red time for the given phase, then there's also no green time for the given phase
                # If (dict_signal_types["red"] == [pd.NaT]) or (dict_signal_types["red"] == [np.nan]):
                if any(pd.isna(time) for time in dict_signal_types["red"]):
                    for signal_type in signal_types:
                        # Add 1 when signal is red, else 0
                        if signal_type == "red":
                            dict_spat_id[f"{signal_type}RatioPhase{phase_no}"] = 1  
                        else:
                            dict_spat_id[f"{signal_type}RatioPhase{phase_no}"] = 0
        
                    # After adding continue (to proceed to next phase in the loop)
                    continue

                # Get green, yellow, and red clearance times
                dict_signal_types["green"] = df_vehicle_cycle_profile_id.loc[i, f"greenPhase{phase_no}"]
                dict_signal_types["yellow"] = df_vehicle_cycle_profile_id.loc[i, f"yellowPhase{phase_no}"]
                dict_signal_types["redClearance"] = df_vehicle_cycle_profile_id.loc[i, f"redClearancePhase{phase_no}"]
        
                # Get and add phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio per cycle
                for signal_type in signal_types:
                    # Check if the instance is not list (if not, convert to list)
                    if not isinstance(dict_signal_types[f"{signal_type}"], list):
                        dict_signal_types[f"{signal_type}"] = [dict_signal_types[f"{signal_type}"]]
                        
                    # Intialize variable to store time difference 
                    time_diff = 0

                    # Loop through signal times (format: [(start, end), (start, end), ..., (start, end)]
                    for start_time, end_time in dict_signal_types[f"{signal_type}"]:
                        # Calculate and store time difference in seconds
                        time_diff += (end_time - start_time).total_seconds()
        
                    # Calcuate and store signal ratio
                    dict_spat_id[f"{signal_type}RatioPhase{phase_no}"] = round(time_diff / cycle_length, 4)
    
            
            # Concatenate phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio per cycle 
            df_spat_id = pd.concat([df_spat_id, pd.DataFrame([dict_spat_id])], axis=0, ignore_index=True)  

        # Add date information to the DataFrame
        df_spat_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

        # Path (from database directory) to directory where cycle-level vehicle signal feature (spat) will be stored
        _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
            resolution_level="cycle",
            event_type="vehicle_signal",
            feature_name="spat",
            signal_id=signal_id
        )

        # Save the phase profile data as a pkl file in the export directory
        export_data(df=df_spat_id, 
                    base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                    sub_dirpath=production_feature_dirpath,
                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                    file_type="pkl")

        return df_spat_id  


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
        self.day = day; self.month = month; self.year = year 

    def _load_data(self, signal_id: str):
        # Path (from database directory) to directory where sorted event data is stored
        interim_event_dirpath, _, _, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(signal_id=signal_id)

        # Load event data
        df_event_id = load_data(base_dirpath=os.path.join(root_dir, relative_interim_database_dirpath),
                                sub_dirpath=interim_event_dirpath, 
                                filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                file_type="pkl")
        
        # Path (from database directory) to directory where signal configuration data is stored
        _, interim_config_dirpath, _, _ = feature_extraction_dirpath.get_feature_extraction_dirpath()

        # Load configuration data for the specific signal
        df_config_id = load_data(base_dirpath=os.path.join(root_dir, relative_interim_database_dirpath), 
                                 sub_dirpath=interim_config_dirpath,
                                 filename=f"{signal_id}",
                                 file_type="csv")
        
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
        
        return df_event_id, df_config_id, df_vehicle_cycle_profile_id

    def extract_volume(self, signal_id: str):
        # Load event, configuration, and vehicle cycle profile data
        df_event_id, df_config_id, df_vehicle_cycle_profile_id = self._load_data(signal_id=signal_id)
        
        # Filter event data for the event code sequence: [81] (i.e., detector off)
        df_event_id = self.filter_by_event_sequence(df_event=df_event_id, 
                                                    event_sequence=[81])

        # Drop rows with missing phase no in configuration data
        df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])].reset_index(drop=True)
        df_config_id = float_to_int(df_config_id) 

        # Filter configuration data for back detector
        df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])]
        df_config_id = df_config_id[df_config_id["stopBarDistance"] != 0].reset_index(drop=True)

        # If multiple back detectors exist, consider farthest of the two detectors from stop bar 
        for phase_no in df_config_id["phaseNo"].unique():
            df_config_phase = df_config_id[df_config_id["phaseNo"] == phase_no]

            back_detectors_at = df_config_phase["stopBarDistance"].unique().tolist()
            if len(back_detectors_at) == 1:
                continue
            else:
                indices = df_config_phase[df_config_phase["stopBarDistance"] == min(back_detectors_at)].index
                indices = indices.tolist()

                df_config_id = df_config_id.drop(index=indices)

        df_config_id = df_config_id.reset_index(drop=True)
        
        # Join configuration data with event data 
        df_event_id = pd.merge(df_event_id, df_config_id, 
                               how="inner", 
                               left_on=["eventParam"], right_on=["channelNo"])

        # Change dtype
        df_event_id = float_to_int(df_event_id)
                
        # Initialize df to store volume data
        df_volume_id = pd.DataFrame()

        for i in range(len(df_vehicle_cycle_profile_id)):
            # Extrac signal info
            signal_id = df_vehicle_cycle_profile_id["signalID"][i] 

            cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
            cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]; cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
            cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]
        
            # Initialize dictionary with signal info to also store phase-specific volume per cycle
            dict_volume_id = {
                "signalID": "0", "cycleNo": 0, "cycleBegin": pd.NaT, "cycleEnd": pd.NaT, "cycleLength": 0
                }

            # List all unique phase nos
            phase_nos = sorted(list(df_event_id["phaseNo"].unique()))

            # Update dictionary to store phase-specific volume per cycle
            for phase_no in phase_nos:
                dict_volume_id.update(
                    {f'volumePhase{phase_no}': 0}
                )
                dict_volume_id.update(
                    {f'greenVolumePhase{phase_no}': 0, 
                     f'yellowVolumePhase{phase_no}': 0, 
                     f'redClearanceVolumePhase{phase_no}': 0, f'redVolumePhase{phase_no}': 0}
                )

            # Add signal info to the dictonary
            dict_volume_id["signalID"] = signal_id; 
            
            dict_volume_id["cycleNo"] = cycle_no
            dict_volume_id["cycleBegin"] = cycle_begin; dict_volume_id["cycleEnd"] = cycle_end
            dict_volume_id["cycleLength"] = cycle_length

            # Filter out all the detector off events within the current cycle, sort, and reset index
            df_event_cycle = df_event_id[(
                (df_event_id["timeStamp"] >= cycle_begin) & 
                (df_event_id["timeStamp"] <= cycle_end)
            )]

            df_event_cycle = df_event_cycle.sort_values(by=["timeStamp", "phaseNo", "channelNo"])
            df_event_cycle = df_event_cycle.reset_index(drop=True)

            for phase_no in phase_nos:
                # Filter event data based on phase no, and reset index
                df_event_phase = df_event_cycle[df_event_cycle["phaseNo"] == phase_no].reset_index(drop=True)

                # Determine phase volume
                phase_volume = len(df_event_phase)

                dict_volume_id[f"volumePhase{phase_no}"] = phase_volume

                # Intialize list of signal types
                signal_types = ["green", "yellow", "redClearance", "red"] 
                
                for signal_type in signal_types:
                    # Get timestamps (format: [(start, end), (start, end), ..., (start, end)]) of the current signal in the current phase
                    timestamps = df_vehicle_cycle_profile_id.loc[i, f'{signal_type}Phase{phase_no}']

                    # Check if the instance is not list (if not, convert to list)
                    if not isinstance(timestamps, list):
                        timestamps = [timestamps]

                    # If the current signal doesn't have any timeslot in the current signal type of the phase, store zero volume
                    if (timestamps == [pd.NaT]) or (all(pd.isna(timestamp) for timestamp in timestamps)):
                        signal_volume = 0
                    else: 
                        signal_volume = 0
                        # Loop through signal times (format: [(start, end), (start, end), ..., (start, end)]
                        for start_time, end_time in timestamps:
                            # calculate and store volume for the signal
                            signal_volume += len(
                                df_event_phase[((df_event_phase["timeStamp"] >= start_time) & (df_event_phase["timeStamp"] <= end_time))]
                                )
                            
                    # Add phase-specific volume for the current signal
                    dict_volume_id[f'{signal_type}VolumePhase{phase_no}'] = signal_volume

            df_volume_id = pd.concat([df_volume_id, pd.DataFrame([dict_volume_id])], 
                                     axis=0, ignore_index=True)
        
        # Add date information to the DataFrame
        df_volume_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

        # Path (from database directory) to directory where cycle-level vehicle traffic feature (volume) will be stored
        _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
            resolution_level="cycle",
            event_type="vehicle_traffic",
            feature_name="volume",
            signal_id=signal_id
        )

        # Save the phase profile data as a pkl file in the export directory
        export_data(df=df_volume_id, 
                    base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                    sub_dirpath=production_feature_dirpath,
                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                    file_type="pkl")

        return df_volume_id
    
    def calculate_stats(self, df: pd.DataFrame, column_names: list, include_sum_list: bool = False):
        """
        Processes specified columns in the DataFrame containing nested lists and generates
        derived statistics columns:
        - Minimum value of the nested list values
        - Maximum value of the nested list values
        - Mean value of the nested list values
        - Standard Deviation of the nested list values
        - Sum of the nested list values (optional)
        The original columns are retained, and the column is flattened in place.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame to process.
        column_names : list
            List of column names in the DataFrame to process.
        include_sum_list : bool, optional
            If True, includes a column for the sum of each nested list. Default is False.

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
                sum_col_name = col.replace(prefix, f"{prefix}SumList")

                # Flatten the original column in place
                dict_stats[col] = df[col].apply(
                    lambda row: [item for lst in row for item in lst if pd.notna(item)] or [0]
                )

                # Calculate the minimum of the nested lists
                dict_stats[min_col_name] = df[col].apply(
                    lambda row: np.nanmin(
                        [np.nanmin(lst) for lst in row if np.nansum(lst) != 0] or [0]
                    )
                )

                # Calculate the maximum of the nested lists
                dict_stats[max_col_name] = df[col].apply(
                    lambda row: np.nanmax(
                        [np.nanmax(lst) for lst in row if np.nansum(lst) != 0] or [0]
                    )
                )

                # Calculate the mean of the flattened list values
                dict_stats[avg_col_name] = df[col].apply(
                    lambda row: np.nanmean(
                        [item for lst in row for item in lst if pd.notna(item)] or [0]
                    )
                )

                # Calculate the standard deviation of the flattened list values
                dict_stats[std_col_name] = df[col].apply(
                    lambda row: np.nanstd(
                        [item for lst in row for item in lst if pd.notna(item)] or [0]
                    )
                )

                # Optionally calculate the sum of each nested list
                if include_sum_list:
                    dict_stats[sum_col_name] = df[col].apply(
                        lambda row: [round(np.nansum(lst), 4) for lst in row if np.nansum(lst) != 0] or [0]
                )

            # Drop the original columns
            df = df.drop(columns=column_names)

            # Use pd.concat to add all derived columns at once
            df = pd.concat([df, pd.DataFrame(dict_stats)], axis=1)

            return df

        except Exception as e:
            raise CustomException(
                custom_message=f"Error processing statistics for columns {column_names}: {str(e)}",
                sys_module=sys
            )

    def extract_occupancy(self, signal_id: str):
        # Load event, configuration, and vehicle cycle profile data
        df_event_id, df_config_id, df_vehicle_cycle_profile_id = self._load_data(signal_id=signal_id)
        
        # Filter event data for the event code sequence: [82, 81]
        df_event_id = self.filter_by_event_sequence(df_event=df_event_id, 
                                                    event_sequence=[82, 81])

        # Drop rows with missing phase no in configuration data
        df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])].reset_index(drop=True)
        df_config_id = float_to_int(df_config_id) 

        # Filter configuration data for stop bar (front detector)
        df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])]
        df_config_id = df_config_id[df_config_id["stopBarDistance"] == 0].reset_index(drop=True)
        
        # Join configuration data with event data 
        df_event_id = pd.merge(df_event_id, df_config_id, 
                               how="inner", 
                               left_on=["eventParam"], right_on=["channelNo"])

        # Change dtype
        df_event_id = float_to_int(df_event_id)
                
        # Initialize df to store occupancy data
        df_occupancy_id = pd.DataFrame()

        for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
            # Extrac signal info
            signal_id = df_vehicle_cycle_profile_id["signalID"][i] 

            cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
            cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]; cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
            cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]
        
            # Initialize dictionary with signal info to also store phase-specific occupancy per cycle
            dict_occupancy_id = {
                "signalID": "0", "cycleNo": 0, "cycleBegin": pd.NaT, "cycleEnd": pd.NaT, "cycleLength": 0
                }

            # List all unique phase nos
            phase_nos = sorted(list(df_event_id["phaseNo"].unique()))

            # Update dictionary to store phase-specific occupancy per cycle
            for phase_no in phase_nos:
                dict_occupancy_id.update(
                    {f"greenOccupancyPhase{phase_no}": [[np.nan]], 
                     f"yellowOccupancyPhase{phase_no}": [[np.nan]], 
                     f"redClearanceOccupancyPhase{phase_no}": [[np.nan]], f"redOccupancyPhase{phase_no}": [[np.nan]]}
                )

            # Add signal info to the dictonary
            dict_occupancy_id["signalID"] = signal_id; 
            
            dict_occupancy_id["cycleNo"] = cycle_no
            dict_occupancy_id["cycleBegin"] = cycle_begin; dict_occupancy_id["cycleEnd"] = cycle_end
            dict_occupancy_id["cycleLength"] = cycle_length

            # Set tolerance of on cycle begin and cycle end to filter, and reset event data: so that every 'on' and 'off' sequence with the current cycle is captured
            if i == 0:  # Check if it is the first index
                st = cycle_begin
                et = df_vehicle_cycle_profile_id.loc[i+1, "cycleEnd"]
            elif i == len(df_vehicle_cycle_profile_id) - 1:  # Check if it is the last index
                st = df_vehicle_cycle_profile_id.loc[i-1, "cycleBegin"]
                et = cycle_end
            else:
                st = df_vehicle_cycle_profile_id.loc[i-1, "cycleBegin"]
                et= df_vehicle_cycle_profile_id.loc[i+1, "cycleEnd"]

            # Filter out all the detector off events within the tolerance, sort, and reset index
            df_event_tolerance = df_event_id[(
                (df_event_id["timeStamp"] >= st) & # st: start time
                (df_event_id["timeStamp"] <= et)   # et: end time
            )]

            df_event_tolerance = df_event_tolerance.sort_values(by=["timeStamp", "phaseNo", "channelNo"])
            df_event_tolerance = df_event_tolerance.reset_index(drop=True)

            for phase_no in phase_nos:
                # Filter event data based on phase no, and reset index
                df_event_phase = df_event_tolerance[df_event_tolerance["phaseNo"] == phase_no].reset_index(drop=True)

                # Intialize list of signal types
                signal_types = ["green", "yellow", "redClearance", "red"] 
                
                for signal_type in signal_types:
                    # Get timestamps (format: [(start, end), (start, end), ..., (start, end)]) of the current signal in the current phase
                    timestamps = df_vehicle_cycle_profile_id.loc[i, f'{signal_type}Phase{phase_no}']

                    # Check if the instance is not list (if not, convert to list)
                    if not isinstance(timestamps, list):
                        timestamps = [timestamps]

                    # If the current signal doesn't have any timeslot in the current signal type of the phase, continue
                    if (timestamps == [pd.NaT]) or (all(pd.isna(timestamp) for timestamp in timestamps)):
                        continue

                    for channel_no in df_event_phase["channelNo"].unique():
                        # Filter events on channel no, sort, and reset index
                        df_event_channel = df_event_phase[df_event_phase["channelNo"] == channel_no]
                        df_event_channel = df_event_channel.sort_values(by="timeStamp").reset_index(drop=True)

                        # Assign sequence ids
                        df_event_channel["sequenceID"] = self.add_event_sequence_id(df_event_channel, 
                                                                                    valid_event_sequence=[82, 81])

                        # Intialize list to append occupancy per cycle of the current signal for the current channel
                        occupancies = [np.nan]
                        
                        # Loop through signal times (format: [(start, end), (start, end), ..., (start, end)]
                        for start_time, end_time in timestamps:
                            for sequence_id in df_event_channel["sequenceID"].unique():
                                # Filter events specific to the current sequence in the given channel, and reset index
                                df_event_sequence = df_event_channel[df_event_channel["sequenceID"] == sequence_id]
                                df_event_sequence = df_event_sequence.reset_index(drop=True)

                                # Continue if the sequence is not complete (either detector on or off is missing)
                                if len(df_event_sequence) != 2:
                                    continue

                                # Extract detector on and off time
                                detector_ont = df_event_sequence["timeStamp"][0] # ont: on time
                                detector_oft = df_event_sequence["timeStamp"][1] # oft: off time
                                    
                                # If detector on and off and start time and end time don't have any common time, continue
                                if ((detector_ont <= start_time) and (detector_oft <= start_time)) or ((detector_ont >= end_time) and (detector_oft >= end_time)):
                                    continue
                                    
                                # Get the intersecting time interval (max, min) of (detector on, detector off) and (signal start time, signal end time)
                                max_st = max(detector_ont, start_time); min_et = min(detector_oft, end_time)

                                # If maximum start time is less than minimum end time, calculate the occupancy for current signal of the current channel (in sec)
                                if max_st < min_et:
                                    # Calculate occupancy for the current signal of the current channel
                                    time_diff = round((min_et - max_st).total_seconds(), 4)

                                    # Append occupancy of the current signal of the current channel
                                    occupancies.append(time_diff)

                        # Append phase-specific occupancy per cycle for the current phase
                        dict_occupancy_id[f'{signal_type}OccupancyPhase{phase_no}'].append(occupancies)
                        
            df_occupancy_id = pd.concat([df_occupancy_id, pd.DataFrame([dict_occupancy_id])],
                                        axis=0, ignore_index=True)
        
        # Add date information to the DataFrame
        df_occupancy_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

        columns = []
        for column in df_occupancy_id.columns:
            if "Occupancy" in column:
                columns.append(column)

        # Calculate stats
        df_occupancy_id = self.calculate_stats(df=df_occupancy_id, column_names=columns,  
                                               include_sum_list=True)


        # Path (from database directory) to directory where cycle-level vehicle traffic feature (occupancy) will be stored
        _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
            resolution_level="cycle",
            event_type="vehicle_traffic",
            feature_name="occupancy",
            signal_id=signal_id
        )

        # Save the phase profile data as a pkl file in the export directory
        export_data(df=df_occupancy_id, 
                    base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                    sub_dirpath=production_feature_dirpath,
                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                    file_type="pkl")

        return df_occupancy_id
    
    def extract_headway(self, signal_id: str):
        # Load event, configuration, and vehicle cycle profile data
        df_event_id, df_config_id, df_vehicle_cycle_profile_id = self._load_data(signal_id=signal_id)
        
        # Filter event data for the event code sequence: [82] (i.e., detector on)
        df_event_id = self.filter_by_event_sequence(df_event=df_event_id, 
                                                    event_sequence=[82])

        # Drop rows with missing phase no in configuration data
        df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])].reset_index(drop=True)
        df_config_id = float_to_int(df_config_id) 

        # Filter configuration data for back detector
        df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])]
        df_config_id = df_config_id[df_config_id["stopBarDistance"] != 0].reset_index(drop=True)

        # If multiple back detectors exist, consider farthest of the two detectors from stop bar 
        for phase_no in df_config_id["phaseNo"].unique():
            df_config_phase = df_config_id[df_config_id["phaseNo"] == phase_no]

            back_detectors_at = df_config_phase["stopBarDistance"].unique().tolist()
            if len(back_detectors_at) == 1:
                continue
            else:
                indices = df_config_phase[df_config_phase["stopBarDistance"] == min(back_detectors_at)].index
                indices = indices.tolist()

                df_config_id = df_config_id.drop(index=indices)

        df_config_id = df_config_id.reset_index(drop=True)
        
        # Join configuration data with event data 
        df_event_id = pd.merge(df_event_id, df_config_id, 
                               how="inner", 
                               left_on=["eventParam"], right_on=["channelNo"])

        # Change dtype
        df_event_id = float_to_int(df_event_id)
                
        # Initialize df to store headway data
        df_headway_id = pd.DataFrame()

        for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
            # Extrac signal info
            signal_id = df_vehicle_cycle_profile_id["signalID"][i] 

            cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
            cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]; cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
            cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]
        
            # Initialize dictionary with signal info to also store phase-specific headway per cycle
            dict_headway_id = {
                "signalID": "0", "cycleNo": 0, "cycleBegin": pd.NaT, "cycleEnd": pd.NaT, "cycleLength": 0
                }

            # List all unique phase nos
            phase_nos = sorted(list(df_event_id["phaseNo"].unique()))

            # Update dictionary to store phase-specific headway per cycle
            for phase_no in phase_nos:
                dict_headway_id.update(
                    {f"greenHeadwayPhase{phase_no}": [[np.nan]], 
                     f"yellowHeadwayPhase{phase_no}": [[np.nan]], 
                     f"redClearanceHeadwayPhase{phase_no}": [[np.nan]], f"redHeadwayPhase{phase_no}": [[np.nan]]}
                )

            # Add signal info to the dictonary
            dict_headway_id["signalID"] = signal_id; 
            
            dict_headway_id["cycleNo"] = cycle_no
            dict_headway_id["cycleBegin"] = cycle_begin; dict_headway_id["cycleEnd"] = cycle_end
            dict_headway_id["cycleLength"] = cycle_length

            # Filter out all the detector off events within the tolerance, sort, and reset index
            df_event_cycle = df_event_id[(
                (df_event_id["timeStamp"] >= cycle_begin) & 
                (df_event_id["timeStamp"] <= cycle_end)  
            )]

            df_event_cycle = df_event_cycle.sort_values(by=["timeStamp", "phaseNo", "channelNo"])
            df_event_cycle = df_event_cycle.reset_index(drop=True)

            for phase_no in phase_nos:
                # Filter event data based on phase no, and reset index
                df_event_phase = df_event_cycle[df_event_cycle["phaseNo"] == phase_no].reset_index(drop=True)

                # Intialize list of signal types
                signal_types = ["green", "yellow", "redClearance", "red"] 
                
                for signal_type in signal_types:
                    # Get timestamps (format: [(start, end), (start, end), ..., (start, end)]) of the current signal in the current phase
                    timestamps = df_vehicle_cycle_profile_id.loc[i, f'{signal_type}Phase{phase_no}']

                    # Check if the instance is not list (if not, convert to list)
                    if not isinstance(timestamps, list):
                        timestamps = [timestamps]

                    # If the current signal doesn't have any timeslot in the current signal type of the phase, continue
                    if (timestamps == [pd.NaT]) or (all(pd.isna(timestamp) for timestamp in timestamps)):
                        continue

                    for channel_no in df_event_phase["channelNo"].unique():
                        # Filter events on channel no, sort, and reset index
                        df_event_channel = df_event_phase[df_event_phase["channelNo"] == channel_no]
                        df_event_channel = df_event_channel.sort_values(by="timeStamp").reset_index(drop=True)

                        # Intialize list to append headway per cycle of the current signal for the current channel
                        headways = [np.nan]
                        
                        # Loop through signal times (format: [(start, end), (start, end), ..., (start, end)]
                        for start_time, end_time in timestamps:
                            # Filter on signal start and end time, sort, and reset index
                            df_event_signal = df_event_channel[((df_event_channel["timeStamp"] >= start_time) & 
                                                                (df_event_channel["timeStamp"] <= end_time))]
                            df_event_signal = df_event_signal.sort_values(by="timeStamp").reset_index(drop=True)

                            # if there's at most one vehicle, there's no headway
                            if len(df_event_signal) <= 1:
                                continue
 
                            for j in range(len(df_event_signal) - 1):
                                # Detector on time when leading and the following (nexy) vehicle passes the back detector
                                detector_ont_lead = df_event_signal["timeStamp"][j] # ont: on time
                                detector_ont_next = df_event_signal["timeStamp"][j+1]

                                # Calculate the difference between detector on times of the two consecutive vehicles (in sec)
                                time_diff = round((detector_ont_next - detector_ont_lead).total_seconds(), 4)

                                headways.append(time_diff)


                        # Append phase-specific headway per cycle for the current phase
                        dict_headway_id[f'{signal_type}HeadwayPhase{phase_no}'].append(headways)
                        
            df_headway_id = pd.concat([df_headway_id, pd.DataFrame([dict_headway_id])],
                                      axis=0, ignore_index=True)
        
        # Add date information to the DataFrame
        df_headway_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

        columns = []
        for column in df_headway_id.columns:
            if "Headway" in column:
                columns.append(column)

        # Calculate stats
        df_headway_id = self.calculate_stats(df=df_headway_id, column_names=columns)

        # Path (from database directory) to directory where cycle-level vehicle traffic feature (headway) will be stored
        _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
            resolution_level="cycle",
            event_type="vehicle_traffic",
            feature_name="headway",
            signal_id=signal_id
        )

        # Save the phase profile data as a pkl file in the export directory
        export_data(df=df_headway_id, 
                    base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                    sub_dirpath=production_feature_dirpath,
                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                    file_type="pkl")

        return df_headway_id
    
    def extract_red_running(self, signal_id: str):
        # Load event, configuration, and vehicle cycle profile data
        df_event_id, df_config_id, df_vehicle_cycle_profile_id = self._load_data(signal_id=signal_id)
        
        # Filter event data for the event code sequence: [81] (i.e., detector off)
        df_event_id = self.filter_by_event_sequence(df_event=df_event_id, 
                                                    event_sequence=[81])
        
        # Drop rows with missing phase no in configuration data
        df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])].reset_index(drop=True)
        df_config_id = float_to_int(df_config_id) 

        # Filter configuration data for stop bar (front detector)
        df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])]
        df_config_id = df_config_id[df_config_id["stopBarDistance"] == 0].reset_index(drop=True)
        
        # Join configuration data with event data 
        df_event_id = pd.merge(df_event_id, df_config_id, 
                               how="inner", 
                               left_on=["eventParam"], right_on=["channelNo"])

        # Change dtype
        df_event_id = float_to_int(df_event_id)
                
        # Initialize df to store red light running data
        df_red_running_id = pd.DataFrame()
        
        for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
            # Extrac signal info
            signal_id = df_vehicle_cycle_profile_id["signalID"][i] 

            cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
            cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]; cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
            cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]
        
            # Initialize dictionary with signal info to also store phase-specific red light running per cycle
            dict_red_running_id = {
                "signalID": "0", "cycleNo": 0, "cycleBegin": pd.NaT, "cycleEnd": pd.NaT, "cycleLength": 0
                }

            # List all unique phase nos
            phase_nos = sorted(list(df_event_id["phaseNo"].unique()))

            # Update dictionary to store phase-specific red light running per cycle
            for phase_no in phase_nos:
                dict_red_running_id.update(
                    {f"redClearanceRunningFlagPhase{phase_no}": 0, f"redRunningFlagPhase{phase_no}": 0}
                )

            # Add signal info to the dictonary
            dict_red_running_id["signalID"] = signal_id; 
            
            dict_red_running_id["cycleNo"] = cycle_no
            dict_red_running_id["cycleBegin"] = cycle_begin; dict_red_running_id["cycleEnd"] = cycle_end
            dict_red_running_id["cycleLength"] = cycle_length

            # Filter out all the detector off events within the tolerance, sort, and reset index
            df_event_cycle = df_event_id[(
                (df_event_id["timeStamp"] >= cycle_begin) & 
                (df_event_id["timeStamp"] <= cycle_end)  
            )]

            df_event_cycle = df_event_cycle.sort_values(by=["timeStamp", "phaseNo", "channelNo"])
            df_event_cycle = df_event_cycle.reset_index(drop=True)

            for phase_no in phase_nos:
                # Filter event data based on phase no, and reset index
                df_event_phase = df_event_cycle[df_event_cycle["phaseNo"] == phase_no].reset_index(drop=True)

                # Intialize list of signal types
                signal_types = ["redClearance", "red"] 
                
                for signal_type in signal_types:
                    # Get timestamps (format: [(start, end), (start, end), ..., (start, end)]) of the current signal in the current phase
                    timestamps = df_vehicle_cycle_profile_id.loc[i, f'{signal_type}Phase{phase_no}']

                    # Check if the instance is not list (if not, convert to list)
                    if not isinstance(timestamps, list):
                        timestamps = [timestamps]

                    # If the current signal doesn't have any timeslot in the current signal type of the phase, continue
                    if (timestamps == [pd.NaT]) or (all(pd.isna(timestamp) for timestamp in timestamps)):
                        continue
                    
                    # Intialize indicator of detecting red light running flag
                    red_running_flag = 0

                    for channel_no in df_event_phase["channelNo"].unique():
                        # Filter events on channel no, sort, and reset index
                        df_event_channel = df_event_phase[df_event_phase["channelNo"] == channel_no]
                        df_event_channel = df_event_channel.sort_values(by="timeStamp").reset_index(drop=True)
                        
                        # Loop through signal times (format: [(start, end), (start, end), ..., (start, end)]
                        for start_time, end_time in timestamps:
                            for j in range(len(df_event_channel)):
                                # Get the time of the detector off event
                                timestamp = df_event_channel.loc[j, 'timeStamp']

                                # If the timestamp for the event is within the start and end time of current signal ('red clearance' or 'red'), there's a red run
                                if (timestamp >= start_time) & (timestamp <= end_time):
                                    red_running_flag = 1
                                    break
                                
                    dict_red_running_id[f'{signal_type}RunningFlagPhase{phase_no}'] = red_running_flag

                        
            df_red_running_id = pd.concat([df_red_running_id, pd.DataFrame([dict_red_running_id])],
                                          axis=0, ignore_index=True)
        
        # Add date information to the DataFrame
        df_red_running_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

        # Path (from database directory) to directory where cycle-level vehicle traffic feature (red running) will be stored
        _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
            resolution_level="cycle",
            event_type="vehicle_traffic",
            feature_name="red_running",
            signal_id=signal_id
        )

        # Save the phase profile data as a pkl file in the export directory
        export_data(df=df_red_running_id, 
                    base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                    sub_dirpath=production_feature_dirpath,
                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                    file_type="pkl")

        return df_red_running_id
        
    def extract_dilemma_running(self, signal_id: str):
        # Load event, configuration, and vehicle cycle profile data
        df_event_id, df_config_id, df_vehicle_cycle_profile_id = self._load_data(signal_id=signal_id)
        
        # Filter event data for the event code sequence: [81] (i.e., detector off)
        df_event_id = self.filter_by_event_sequence(df_event=df_event_id, 
                                                    event_sequence=[81])
        
        # Drop rows with missing phase no in configuration data
        df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])].reset_index(drop=True)
        df_config_id = float_to_int(df_config_id) 

        # Filter configuration data for stop bar (front detector)
        df_config_id = df_config_id[pd.notna(df_config_id["phaseNo"])]
        df_config_id = df_config_id[df_config_id["stopBarDistance"] == 0].reset_index(drop=True)
        
        # Join configuration data with event data 
        df_event_id = pd.merge(df_event_id, df_config_id, 
                               how="inner", 
                               left_on=["eventParam"], right_on=["channelNo"])

        # Change dtype
        df_event_id = float_to_int(df_event_id)
                
        # Initialize df to store dilemma zone running data
        df_dilemma_running_id = pd.DataFrame()

        for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
            # Extrac signal info
            signal_id = df_vehicle_cycle_profile_id["signalID"][i] 

            cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
            cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]; cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
            cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]
        
            # initialize dictionary
            dict_dilemma_running_id = {
                "signalID": "0", "cycleNo": 0, "cycleBegin": pd.NaT, "cycleEnd": pd.NaT, "cycleLength": 0
                }

            # List all unique phase nos
            phase_nos = sorted(list(df_event_id["phaseNo"].unique()))

            # Update dictionary to store phase-specific dilemma zone running per cycle
            for phase_no in phase_nos:
                dict_dilemma_running_id.update(
                    {f"yellowRunningFlagPhase{phase_no}": 0}
                )

            # Add signal info to the dictonary
            dict_dilemma_running_id["signalID"] = signal_id; 
            
            dict_dilemma_running_id["cycleNo"] = cycle_no
            dict_dilemma_running_id["cycleBegin"] = cycle_begin; dict_dilemma_running_id["cycleEnd"] = cycle_end
            dict_dilemma_running_id["cycleLength"] = cycle_length

            # Filter out all the detector off events within the tolerance, sort, and reset index
            df_event_cycle = df_event_id[(
                (df_event_id["timeStamp"] >= cycle_begin) & 
                (df_event_id["timeStamp"] <= cycle_end)  
            )]

            df_event_cycle = df_event_cycle.sort_values(by=["timeStamp", "phaseNo", "channelNo"])
            df_event_cycle = df_event_cycle.reset_index(drop=True)

            for phase_no in phase_nos:
                # Filter event data based on phase no, and reset index
                df_event_phase = df_event_cycle[df_event_cycle["phaseNo"] == phase_no].reset_index(drop=True)

                # Intialize list of signal types
                signal_types = ["yellow"] 
                
                for signal_type in signal_types:
                    # Get timestamps (format: [(start, end), (start, end), ..., (start, end)]) of the current signal in the current phase
                    timestamps = df_vehicle_cycle_profile_id.loc[i, f'{signal_type}Phase{phase_no}']

                    # Check if the instance is not list (if not, convert to list)
                    if not isinstance(timestamps, list):
                        timestamps = [timestamps]

                    # If the current signal doesn't have any timeslot in the current signal type of the phase, continue
                    if (timestamps == [pd.NaT]) or (all(pd.isna(timestamp) for timestamp in timestamps)):
                        continue
                    
                    # Intialize indicator of detecting dilemma zone running flag
                    dilemma_running_flag = 0

                    for channel_no in df_event_phase["channelNo"].unique():
                        # Filter events on channel no, sort, and reset index
                        df_event_channel = df_event_phase[df_event_phase["channelNo"] == channel_no]
                        df_event_channel = df_event_channel.sort_values(by="timeStamp").reset_index(drop=True)
                        
                        # Loop through signal times (format: [(start, end), (start, end), ..., (start, end)]
                        for start_time, end_time in timestamps:
                            for j in range(len(df_event_channel)):
                                # Get the time of the detector off event
                                timestamp = df_event_channel.loc[j, 'timeStamp']

                                # If the timestamp for the event is within the start and end time of current signal ('red clearance' or 'red'), there's a red run
                                if (timestamp >= start_time) & (timestamp <= end_time):
                                    dilemma_running_flag = 1
                                    break
                                
                    dict_dilemma_running_id[f'{signal_type}RunningFlagPhase{phase_no}'] = dilemma_running_flag

                        
            df_dilemma_running_id = pd.concat([df_dilemma_running_id, pd.DataFrame([dict_dilemma_running_id])],
                                              axis=0, ignore_index=True)
        
        # Add date information to the DataFrame
        df_dilemma_running_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

        # Path (from database directory) to directory where cycle-level vehicle traffic feature (dilemma running) will be stored
        _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
            resolution_level="cycle",
            event_type="vehicle_traffic",
            feature_name="dilemma_running",
            signal_id=signal_id
        )

        # Save the phase profile data as a pkl file in the export directory
        export_data(df=df_dilemma_running_id, 
                    base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                    sub_dirpath=production_feature_dirpath,
                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                    file_type="pkl")

        return df_dilemma_running_id








