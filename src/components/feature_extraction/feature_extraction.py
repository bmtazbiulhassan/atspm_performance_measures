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
        pd.DataFrame
            DataFrame with an added 'sequenceID' column.

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

            # Add sequenceID column to DataFrame
            df_event["sequenceID"] = sequence_ids

            return df_event
        
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
                df_event_phase = self.add_event_sequence_id(df_event_phase, valid_event_sequence=valid_event_sequence)

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
                signal_id=signal_id, 
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
                signal_id=signal_id, 
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
                signal_id=signal_id, 
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
                signal_id=signal_id, 
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
                signal_id=signal_id, 
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
                signal_id=signal_id, 
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
        

class SPATFeatureExtract(CoreEventUtils):

    def __init__(self):
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

    def spat(self, signal_id: str):
        # Path (from database directory) to directory where cycle-level vehicle signal profile is stored
        _, _, production_signal_dirpath, _ = feature_extraction_dirpath.get_feature_extraction_dirpath(
            signal_id=signal_id, 
            resolution_level="cycle",
            event_type="vehicle_signal",
            signal_id=signal_id
        )
        
        # Load vehicle cycle profile data 
        df_vehicle_cycle_profile_id = load_data(base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                                                sub_dirpath=production_signal_dirpath, 
                                                filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                                file_type="pkl")
        # list of signals
        signal_types = ['green', 'yellow', 'redClearance', 'red']
        
        # initialize dataframe to store phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio per cycle
        df_spat_id = pd.DataFrame() # stratio: signal (i.e., 'green', 'yellow', 'redClearance', and 'red') ratio
        
        # loop through each cycle to calculate and update phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio per cycle
        for i in range(len(df_vehicle_cycle_profile_id)):
            # extract signal info: cycle
            signal_id = df_vehicle_cycle_profile_id.signalID[i]; cycle_no = df_vehicle_cycle_profile_id.cycleNo[i]
            cycle_begin = df_vehicle_cycle_profile_id.cycleBegin[i]; cycle_end = df_vehicle_cycle_profile_id.cycleEnd[i]
            cycle_length = df_vehicle_cycle_profile_id.cycleLength[i]
        
            # initialize a dictionary with signal info: cycle
            dict_spat = {'signalID': '0', 'cycleNo': 0, 'cycleBegin': pd.NaT, 'cycleEnd': pd.NaT, 'cycleLength': 0} 
        
            # get list of unique phase nos from columns
            phase_nos = [int(column[-1]) for column in df_vehicle_cycle_profile_id.columns if "Phase" in column]
        
            # update dictionary to hold phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio per cycle
            for phase_no in phase_nos:
                dict_spat.update(
                    {f'greenRatioPhase{phase_no}': 0, 
                     f'yellowRatioPhase{phase_no}': 0, 
                     f'redClearanceRatioPhase{phase_no}': 0, f'redRatioPhase{phase_no}': 0}
                )
        
            # add signal info: cycle
            dict_spat['signalID'] = signal_id; dict_spat['cycleNo'] = cycle_no
            dict_spat['cycleBegin'] = cycle_begin; dict_spat['cycleEnd'] = cycle_end
            dict_spat['cycleLength'] = cycle_length
        
            # intialize dictionary to temporarily add signal times of every phase
            dict_signal_types = {'green': [], 'yellow': [], 'redClearance': [], 'red': []}
            
            # loop through phase nos to extract phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio
            for phase_no in phase_nos:
                # get red times
                dict_signal_types['red'] = df_vehicle_cycle_profile_id.loc[i, f'redPhase{phase_no}']

                # check if the instance is not list (if not, convert to list)
                if not isinstance(dict_signal_types['red'], list):
                    dict_signal_types['red'] = [dict_signal_types['red']]
        
                # if there's no red time for the given phase, then there's also no green time for the given phase
                if (dict_signal_types['red'] == [pd.NaT]) or (dict_signal_types['red'] == [np.nan]):
                    # loop through signals to add phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio per cycle
                    for signal_type in signal_types:
                        # add 1 when signal is red, else 0
                        if signal_type == 'red':
                            dict_spat[f'{signal_type}RatioPhase{phase_no}'] = 1  
                        else:
                            dict_spat[f'{signal_type}RatioPhase{phase_no}'] = 0
        
                    # after adding continue (to proceed to next phase in the loop)
                    continue
        
                
                # get green, yellow, and red clearance times
                dict_signal_types['green'] = df_vehicle_cycle_profile_id.loc[i, f'greenPhase{phase_no}']
                dict_signal_types['yellow'] = df_vehicle_cycle_profile_id.loc[i, f'yellowPhase{phase_no}']
                dict_signal_types['redClearance'] = df_vehicle_cycle_profile_id.loc[i, f'redClearancePhase{phase_no}']
        
                # get and add phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio per cycle
                for signal_type in signal_types:
                    # check if the instance is not list (if not, convert to list)
                    if not isinstance(dict_signal_types[f'{signal_type}'], list):
                        dict_signal_types[f'{signal_type}'] = [dict_signal_types[f'{signal_type}']]
                        
                    # intialize variable to store time difference 
                    time_diff = 0

                    # loop through signal times (format: [(start, end), (start, end), ..., (start, end)]
                    for start_time, end_time in dict_signal_types[f'{signal_type}']:
                        # calculate and store time difference in seconds
                        time_diff += (end_time - start_time).total_seconds()
        
                    # calcuate and store signal ratio
                    dict_spat[f'{signal_type}RatioPhase{phase_no}'] = round(time_diff / cycle_length, 4)
        
            
            # concatenate phase-specific green ratio, yellow ratio, red clearance ratio, and red ratio per cycle 
            df_spat_id = pd.concat([df_spat_id, pd.DataFrame([dict_spat])], axis=0, ignore_index=True)  


        # Path (from database directory) to directory where cycle-level vehicle signal feature (spat) will be stored
        _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
            signal_id=signal_id, 
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

    def red_light_running(self, signal_id: str):
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
            signal_id=signal_id, 
            resolution_level="cycle",
            event_type="vehicle_signal",
            signal_id=signal_id
        )
        
        # Load vehicle cycle profile data 
        df_vehicle_cycle_profile_id = load_data(base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                                                sub_dirpath=production_signal_dirpath, 
                                                filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                                file_type="pkl")
        
        # Filter event data for event codes 81: detector 'off'
        df_event_id = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=[81])

        # Get column names
        dict_column_names = {
            "count": get_column_name_by_partial_name(df=df_config_id, partial_name="count"),
            "movement": get_column_name_by_partial_name(df=df_config_id, partial_name="movement"),

            "phase": get_column_name_by_partial_name(df=df_config_id, partial_name="phase"),
            "channel": get_column_name_by_partial_name(df=df_config_id, partial_name="count"),
            "signalID": get_column_name_by_partial_name(df=df_config_id, partial_name="signal"),
            
            "time": get_column_name_by_partial_name(df=df_event_id, partial_name="time"),
            "param": get_column_name_by_partial_name(df=df_event_id, partial_name="param")
            }

        # Keep detector info for only the count bar 
        df_config_id = df_config_id[df_config_id[dict_column_names["count"]] == 1]
        df_config_id = df_config_id[df_config_id[dict_column_names["movement"]].isin(["L", "T"])].reset_index(drop=True)
        
        # Join detector info with event data 
        df_event_id = pd.merge(df_event_id, df_config_id, 
                               how="left", 
                               left_on=[[dict_column_names["signalID"]], dict_column_names["param"]], 
                               right_on=[dict_column_names["signalID"], dict_column_names["channel"]])
        
        # Keep observations when channel no is not na, and reset index
        df_event_id = df_event_id[pd.notna(df_event_id[dict_column_names["channel"]])]
        df_event_id = df_event_id.reset_index(drop=True)

        # change dtype
        df_event_id = float_to_int(df_event_id)
                
        # initialize df to store data
        df_red_light_running_id = pd.DataFrame()

        # loop through all cycles
        for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
            # extrac signal info: cycle
            signal_id = df_vehicle_cycle_profile_id["signalID"][i] 
            cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
            cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]
            cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
            cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]
        
            # initialize dictionary
            dict_red_light_running_id = {
                "signalID": "0", "cycleNo": 0, "cycleBegin": pd.NaT, "cycleEnd": pd.NaT, "cycleLength": 0
                }

            # list all unique phase nos
            phase_nos = sorted(list(df_event_id[dict_column_names["phase"]].unique()))

            # add features to dictionary
            for phase_no in phase_nos:
                dict_red_light_running_id.update(
                    {f"redClearanceRunCntPhase{phase_no}": [], f"redClearanceRunFlagPhase{phase_no}": 0,
                    f"redRunCntPhase{phase_no}": [], f"redRunFlagPhase{phase_no}": 0}
                )

            # filter out all the detector off events within the current cycle, sort, and reset index
            df_event_cycle = df_event_id[(
                (df_event_id[dict_column_names["time"]] >= cycle_begin) & 
                (df_event_id[dict_column_names["time"]] <= cycle_end)
            )]

            df_event_cycle = df_event_cycle.sort_values(by=[dict_column_names["time"], 
                                                            dict_column_names["phase"], 
                                                            dict_column_names["channel"]])
            df_event_cycle = df_event_cycle.reset_index(drop=True)

            # loop through each phase to determine red light running during red clearance and red signals
            for phase_no in phase_nos:
                # filter events specific to the current phase in the given cycle, and reset index
                df_event_phase = df_event_cycle[df_event_cycle[dict_column_names["phase"]] == phase_no]
                df_event_phase = df_event_phase.reset_index(drop=True)

                # list all unique channels in the current phase
                channel_nos = list(df_event_phase[dict_column_names["channel"]].unique())

                # intialize dictionary of columns on the red clearance, and red signal timings of the current phase
                dict_signal_columns = {
                    "redClearance": f"redClearancePhase{phase_no}", "red": f"redPhase{phase_no}"
                } 

                # calculate phase-specific determine red light running during red clearance and red signals based on red clearance, and red signal times of the current phase, and detector off times of the current channel
                # note: all channels in a given phase experiences the same signal times
                for key in dict_signal_columns.keys(): # key represents signal (green, yellow, red clearance, and red)
                    # check if the signal phase time is null
                    if df_vehicle_cycle_profile_id.loc[i, dict_signal_columns[key]] == [pd.NaT]:
                        continue

                    # continue if the value is not list i.e., is NaN
                    if not isinstance(df_vehicle_cycle_profile_id.loc[i, dict_signal_columns[key]], list): # pd.isna(df_vehicle_cycle_profile_id.loc[i, dict_columns[key]])
                        continue
                    
                    # loop through channels 
                    for channel_no in channel_nos:
                        # filter events specific to the current channel in the current phase, sort, and reset index
                        df_event_channel = df_event_phase[df_event_phase[dict_column_names["channel"]] == channel_no]
                        df_event_channel = df_event_channel.sort_values(by="timeStamp").reset_index(drop=True)
                        
                        # initialize variable to store red light running count value for the current channel
                        red_light_running_count = 0
                                
                        # loop through the signal times
                        for time_stamps in df_red_light_running_id.loc[i, dict_signal_columns[key]]: # time_stamps format: [(signal start time, signal end time), ......, (signal start time, signal end time)]
                            # extract start and end signal time 
                            start_time = time_stamps[0]; end_time = time_stamps[-1]

                            # through through each detector off event for the current channel
                            for j in range(df_event_channel.shape[0]):
                                # get the time of the detector off event
                                time_stamp = df_event_channel.loc[j, "timeStamp"]

                                # if the timestamp for the event is within the start and end time of current signal ("red clearance" or "red"), there"s a red run
                                if (time_stamp >= start_time) & (time_stamp <= end_time):
                                    red_light_running_count += 1
                                
                        dict_red_light_running_id[f"{key}RunCntPhase{phase_no}"].append(red_light_running_count)

                        # check if all values are not 0, i.e., red running 
                        if not all(value == 0 for value in dict_red_light_running_id[f"{key}RunCntPhase{phase_no}"]):
                            dict_red_light_running_id[f"{key}RunFlagPhase{phase_no}"] = 1
                            

            # update dictionary
            dict_red_light_running_id["signalID"] = signal_id; dict_red_light_running_id["cycleNo"] = cycle_no
            dict_red_light_running_id["cycleBegin"] = cycle_begin; dict_red_light_running_id["cycleEnd"] = cycle_end
            dict_red_light_running_id["cycleLength"] = cycle_length
        
            df_red_light_running_id = pd.concat([df_red_light_running_id, pd.DataFrame([dict_red_light_running_id])], 
                                                 axis=0, ignore_index=True)
        
        # Sort the resulting DataFrame by cycle number and phase number
        df_pedestrian_cycle_profile_id = df_pedestrian_cycle_profile_id.sort_values(by=["cycleNo"]).reset_index(drop=True)

        # Add date information to the DataFrame
        df_red_light_running_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

        # Revise signal export sub-directories
        feature_export_sub_dirs = self.feature_export_sub_dirs+ ["red_light_running", f"{signal_id}"]

        # Path (from database directory) to directory where cycle-level vehicle traffic feature (red light running) will be stored
        _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
            signal_id=signal_id, 
            resolution_level="cycle",
            event_type="vehicle_traffic",
            feature_name="red_light_running",
            signal_id=signal_id
        )

        # Save the phase profile data as a pkl file in the export directory
        export_data(df=df_red_light_running_id, 
                    base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                    sub_dirpath=production_feature_dirpath,
                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                    file_type="pkl")

        return df_red_light_running_id
    
    def dilemma_zone_running(self, signal_id: str):
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
            signal_id=signal_id, 
            resolution_level="cycle",
            event_type="vehicle_signal",
            signal_id=signal_id
        )
        
        # Load vehicle cycle profile data 
        df_vehicle_cycle_profile_id = load_data(base_dirpath=os.path.join(root_dir, relative_production_database_dirpath),
                                                sub_dirpath=production_signal_dirpath, 
                                                filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                                file_type="pkl")
        # Filter event data for event codes 81: detector 'off'
        df_event_id = self.filter_by_event_sequence(df_event=df_event_id, event_sequence=[81])

        # Get column names
        dict_column_names = {
            "count": get_column_name_by_partial_name(df=df_config_id, partial_name="count"),
            "movement": get_column_name_by_partial_name(df=df_config_id, partial_name="movement"),

            "phase": get_column_name_by_partial_name(df=df_config_id, partial_name="phase"),
            "channel": get_column_name_by_partial_name(df=df_config_id, partial_name="count"),
            "signalID": get_column_name_by_partial_name(df=df_config_id, partial_name="signal"),
            
            "time": get_column_name_by_partial_name(df=df_event_id, partial_name="time"),
            "param": get_column_name_by_partial_name(df=df_event_id, partial_name="param")
            }

        # Keep detector info for only the count bar 
        df_config_id = df_config_id[df_config_id[dict_column_names["count"]] == 1]
        df_config_id = df_config_id[df_config_id[dict_column_names["movement"]].isin(["L", "T"])].reset_index(drop=True)
        
        # Join detector info with event data 
        df_event_id = pd.merge(df_event_id, df_config_id, 
                               how="left", 
                               left_on=[[dict_column_names["signalID"]], dict_column_names["param"]], 
                               right_on=[dict_column_names["signalID"], dict_column_names["channel"]])
        
        # Keep observations when channel no is not na, and reset index
        df_event_id = df_event_id[pd.notna(df_event_id[dict_column_names["channel"]])]
        df_event_id = df_event_id.reset_index(drop=True)

        # change dtype
        df_event_id = float_to_int(df_event_id)
                
        # initialize df to store data
        df_dilemma_zone_running_id = pd.DataFrame()

        # loop through all cycles
        for i in tqdm.tqdm(range(len(df_vehicle_cycle_profile_id))):
            # extrac signal info: cycle
            signal_id = df_vehicle_cycle_profile_id["signalID"][i] 
            cycle_no = df_vehicle_cycle_profile_id["cycleNo"][i]
            cycle_begin = df_vehicle_cycle_profile_id.loc[i, "cycleBegin"]
            cycle_end = df_vehicle_cycle_profile_id.loc[i, "cycleEnd"]
            cycle_length = df_vehicle_cycle_profile_id.loc[i, "cycleLength"]
        
            # initialize dictionary
            dict_dilemma_zone_running_id = {
                "signalID": "0", "cycleNo": 0, "cycleBegin": pd.NaT, "cycleEnd": pd.NaT, "cycleLength": 0
                }

            # list all unique phase nos
            phase_nos = sorted(list(df_event_id[dict_column_names["phase"]].unique()))

            # add features to dictionary
            for phase_no in phase_nos:
                dict_dilemma_zone_running_id.update(
                    {f"yellowRunCntPhase{phase_no}": [], f"yellowRunCntPhase{phase_no}": []}
                    )

            # filter out all the detector off events within the current cycle, sort, and reset index
            df_event_cycle = df_event_id[(
                (df_event_id[dict_column_names["time"]] >= cycle_begin) & 
                (df_event_id[dict_column_names["time"]] <= cycle_end)
            )]

            df_event_cycle = df_event_cycle.sort_values(by=[dict_column_names["time"], 
                                                            dict_column_names["phase"], 
                                                            dict_column_names["channel"]])
            df_event_cycle = df_event_cycle.reset_index(drop=True)

            # loop through each phase to determine red light running during red clearance and red signals
            for phase_no in phase_nos:
                # filter events specific to the current phase in the given cycle, and reset index
                df_event_phase = df_event_cycle[df_event_cycle[dict_column_names["phase"]] == phase_no]
                df_event_phase = df_event_phase.reset_index(drop=True)

                # list all unique channels in the current phase
                channel_nos = list(df_event_phase[dict_column_names["channel"]].unique())

                # intialize dictionary of columns on the red clearance, and red signal timings of the current phase
                dict_signal_columns = {
                    "yellow": f"yellowPhase{phase_no}"
                } 

                # calculate phase-specific determine red light running during red clearance and red signals based on red clearance, and red signal times of the current phase, and detector off times of the current channel
                # note: all channels in a given phase experiences the same signal times
                for key in dict_signal_columns.keys(): # key represents signal (green, yellow, red clearance, and red)
                    # check if the signal phase time is null
                    if df_vehicle_cycle_profile_id.loc[i, dict_signal_columns[key]] == [pd.NaT]:
                        continue

                    # continue if the value is not list i.e., is NaN
                    if not isinstance(df_vehicle_cycle_profile_id.loc[i, dict_signal_columns[key]], list): # pd.isna(df_vehicle_cycle_profile_id.loc[i, dict_columns[key]])
                        continue
                    
                    # loop through channels 
                    for channel_no in channel_nos:
                        # filter events specific to the current channel in the current phase, sort, and reset index
                        df_event_channel = df_event_phase[df_event_phase[dict_column_names["channel"]] == channel_no]
                        df_event_channel = df_event_channel.sort_values(by="timeStamp").reset_index(drop=True)
                        
                        # initialize variable to store red light running count value for the current channel
                        dilemma_zone_running_count = 0
                                
                        # loop through the signal times
                        for time_stamps in df_dilemma_zone_running_id.loc[i, dict_signal_columns[key]]: # time_stamps format: [(signal start time, signal end time), ......, (signal start time, signal end time)]
                            # extract start and end signal time 
                            start_time = time_stamps[0]; end_time = time_stamps[-1]

                            # through through each detector off event for the current channel
                            for j in range(df_event_channel.shape[0]):
                                # get the time of the detector off event
                                time_stamp = df_event_channel.loc[j, "timeStamp"]

                                # if the timestamp for the event is within the start and end time of current signal ("red clearance" or "red"), there"s a red run
                                if (time_stamp >= start_time) & (time_stamp <= end_time):
                                    dilemma_zone_running_count += 1
                                
                        dict_dilemma_zone_running_id[f"{key}RunCntPhase{phase_no}"].append(dilemma_zone_running_count)

                        # check if all values are not 0, i.e., red running 
                        if not all(value == 0 for value in dict_dilemma_zone_running_id[f"{key}RunCntPhase{phase_no}"]):
                            dict_dilemma_zone_running_id[f"{key}RunFlagPhase{phase_no}"] = 1
                            

            # update dictionary
            dict_dilemma_zone_running_id["signalID"] = signal_id; dict_dilemma_zone_running_id["cycleNo"] = cycle_no
            dict_dilemma_zone_running_id["cycleBegin"] = cycle_begin; dict_dilemma_zone_running_id["cycleEnd"] = cycle_end
            dict_dilemma_zone_running_id["cycleLength"] = cycle_length
        
            df_dilemma_zone_running_id = pd.concat([df_dilemma_zone_running_id, pd.DataFrame([dict_dilemma_zone_running_id])], 
                                                   axis=0, ignore_index=True)
        
        # Sort the resulting DataFrame by cycle number and phase number
        df_pedestrian_cycle_profile_id = df_pedestrian_cycle_profile_id.sort_values(by=["cycleNo"]).reset_index(drop=True)

        # Add date information to the DataFrame
        df_dilemma_zone_running_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

        # Path (from database directory) to directory where cycle-level vehicle traffic feature (dilemma zone running) will be stored
        _, _, _, production_feature_dirpath = feature_extraction_dirpath.get_feature_extraction_dirpath(
            signal_id=signal_id, 
            resolution_level="cycle",
            event_type="vehicle_traffic",
            feature_name="dilemma_zone_running",
            signal_id=signal_id
        )

        # Save the phase profile data as a pkl file in the export directory
        export_data(df=df_dilemma_zone_running_id, 
                    base_dirpath=os.path.join(root_dir, relative_production_database_dirpath), 
                    sub_dirpath=production_feature_dirpath,
                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                    file_type="pkl")

        return df_dilemma_zone_running_id








