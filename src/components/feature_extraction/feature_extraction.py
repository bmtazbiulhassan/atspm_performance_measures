import pandas as pd
import yaml
import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import get_root_directory, get_column_name_by_partial_name, float_to_int, load_data, export_data


# Get the root directory of the project
root_dir = get_root_directory()

# Load the YAML configuration file (using absolute path)
with open(os.path.join(root_dir, "config.yaml"), "r") as file:
    config = yaml.safe_load(file)

# Retrive relative base dir
relative_interim_base_dir = config["relative_base_dir"]["interim"]
relative_production_base_dir = config["relative_base_dir"]["production"]

# Retrieve settings for feature extraction
config = config["feature_extraction"]


class CoreEventUtils:
    
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
        
        # Retrieve event import and signal export sub-directories
        self.event_import_sub_dirs = config["sunstore"]["event_import_sub_dirs"]
        self.signal_export_sub_dirs = config["sunstore"]["signal_export_sub_dirs"]

        # Retrieve config import sub-directories
        self.config_import_sub_dirs = config["noemi"]["event_import_sub_dirs"]

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
        # Load configuration data for the specific signal
        df_config_id = load_data(base_dir=os.path.join(root_dir, relative_interim_base_dir), 
                                 filename=f"{signal_id}",
                                 file_type="csv", 
                                 sub_dirs=self.config_import_sub_dirs)

        # Convert columns with float values to integer type where applicable
        df_config_id = float_to_int(df_config_id)

        # Dynamically fetch the phase column name based on partial match
        dict_column_names = {"phase": get_column_name_by_partial_name(df=df_config_id, partial_name="phase")}
        phase_nos = df_config_id[dict_column_names["phase"]].unique().tolist()

        # Check if phase numbers in configuration data have corresponding entries in barrier map
        if all(phase_no not in config["noemi"]["barrier_map"].keys() for phase_no in phase_nos):
            raise CustomException(
                custom_message=f"Barrier map {config["noemi"]["barrier_map"]} is not valid for signal ID {signal_id}", 
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
        # Revise event import sub-directories based on signal id
        event_import_sub_dirs = self.event_import_sub_dirs + [f"{signal_id}"]

        # Set event sequence and mapping based on signal type
        if event_type == "vehicle_signal":
            # Mapping event codes to respective phase times
            event_code_map = {
                "greenBegin": 1, "greenEnd": 8,
                "yellowBegin": 8, "yellowEnd": 10,
                "redClearanceBegin": 10, "redClearanceEnd": 11
            }
            valid_event_sequence = config["sunstore"]["valid_event_sequence"]["vehicle_signal"]

        elif event_type == "pedestrian_signal":
            # Mapping event codes for pedestrian phases
            event_code_map = {
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

            # Load event data
            df_event_id = load_data(base_dir=os.path.join(root_dir, relative_interim_base_dir), 
                                    filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                                    file_type="pkl", 
                                    sub_dirs=event_import_sub_dirs)

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
                           if code in current_event_sequence else pd.NaT for key, code in event_code_map.items()}
                    }
                    
                    # Conditionally add "barrierNo" if the signal type is "vehicle"
                    if event_type == "vehicle":
                        dict_phase_profile_id["barrierNo"] = config["noemi"]["barrier_map"].get(int(phase_no), 0)

                    phase_profile_id.append(dict_phase_profile_id)

            # Convert phase profile to DataFrame 
            df_phase_profile_id = pd.DataFrame(phase_profile_id)

            # Create a pseudo timestamp for sorting
            time_columns = [column for column in df_phase_profile_id.columns if column.endswith("Begin") or column.endswith("End")]
            df_phase_profile_id["pseudoTimestamp"] = df_phase_profile_id[time_columns].bfill(axis=1).iloc[:, 0]
            df_phase_profile_id = df_phase_profile_id.sort_values(by="pseudoTimeStamp").reset_index(drop=True)
            df_phase_profile_id.drop(columns=["pseudoTimeStamp"], inplace=True)

            # Add date information
            df_phase_profile_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

            # Revise signal export sub-directories
            signal_export_sub_dirs = self.signal_export_sub_dirs + ["phase" + f"{event_type}", f"{signal_id}"]

            # Save the phase profile data as a pkl file in the export directory
            export_data(df=df_phase_profile_id, 
                        base_dir=os.path.join(root_dir, relative_production_base_dir), 
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                        file_type="pkl", 
                        sub_dirs=signal_export_sub_dirs)

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
            # Define the file path for the vehicle phase data of the specified signal and date
            profile_filepath = os.path.join(
                root_dir, relative_production_base_dir, *self.signal_export_sub_dirs, "phase", "vehicle_signal", signal_id,
                f"{self.year}-{self.month:02d}-{self.day:02d}.pkl"
            )

            # Load vehicle phase profile data or extract it if file does not exist
            if os.path.exists(profile_filepath):
                df_vehicle_phase_profile_id = pd.read_pickle(profile_filepath)
            else:
                df_vehicle_phase_profile_id = self.extract_vehicle_phase_profile(signal_id=signal_id)

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

                    # Initialize phase time columns with NaT for the current phase
                    dict_vehicle_cycle_profile_id.update(
                        {f"{signal_time_type}Phase{phase_no}": [pd.NaT] for signal_time_type in ["green", "yellow", "redClearance", "red"]}
                    )

                    # Initialize dictionary to store signal times for each phase
                    dict_signal_times = {
                        signal_time_type: [] for signal_time_type in ["green", "yellow", "redClearance", "red"]
                    }

                    # Collect start and end times for green, yellow, and redClearance for each phase 
                    for idx in range(df_vehicle_phase_profile_phase.shape[0]):
                        dict_signal_times["green"].append(
                            tuple([df_vehicle_phase_profile_phase.loc[idx, "greenBegin"], df_vehicle_phase_profile_phase.loc[idx, "greenEnd"]])
                            )
                        dict_signal_times["yellow"].append(
                            tuple([df_vehicle_phase_profile_phase.loc[idx, "yellowBegin"], df_vehicle_phase_profile_phase.loc[idx, "yellowEnd"]])
                            )
                        dict_signal_times["redClearance"].append(
                            tuple([df_vehicle_phase_profile_phase.loc[idx, "redClearanceBegin"], df_vehicle_phase_profile_phase.loc[idx, "redClearanceEnd"]])
                            )

                    # Sort all phase time intervals in order
                    signal_times = [tuple([cycle_begin])] + dict_signal_times["green"] + dict_signal_times["yellow"] + dict_signal_times["redClearance"] + [tuple([cycle_end])]
                    signal_times = sorted(signal_times, key=lambda x: x[0])
                    
                    # Generate 'red' intervals by identifying gaps between sorted intervals
                    for start, end in zip(signal_times[:-1], signal_times[1:]):
                        if start[-1] == end[0]:
                            continue
                        dict_signal_times["red"].append((start[-1], end[0]))

                    # Update cycle information dictionary with collected signal times for each type
                    for signal_time_type in ["green", "yellow", "redClearance", "red"]:
                        dict_vehicle_cycle_profile_id[f"{signal_time_type}Phase{phase_no}"] = dict_signal_times[signal_time_type]

                # Append the current cycle information to the cycle profile DataFrame
                df_vehicle_cycle_profile_id = pd.concat([df_vehicle_cycle_profile_id, pd.DataFrame([dict_vehicle_cycle_profile_id])], 
                                                        ignore_index=True)

            # Sort cycle profiles and drop incomplete first and last cycles
            df_vehicle_cycle_profile_id = df_vehicle_cycle_profile_id.sort_values(by=["cycleNo"]).iloc[1:-1].reset_index(drop=True)
            
            # Add date information to the DataFrame
            df_vehicle_cycle_profile_id["date"] = pd.Timestamp(f"{self.year}-{self.month}-{self.day}").date()

            # Revise signal export sub-directories
            signal_export_sub_dirs = self.signal_export_sub_dirs + ["cycle" + "vehicle_signal", f"{signal_id}"]

            # Save the phase profile data as a pkl file in the export directory
            export_data(df=df_vehicle_cycle_profile_id, 
                        base_dir=os.path.join(root_dir, relative_production_base_dir), 
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                        file_type="pkl", 
                        sub_dirs=signal_export_sub_dirs)

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
            # Define the file path for the pedestrian phase data of the specified signal and date
            profile_filepath = os.path.join(
                root_dir, relative_production_base_dir, *self.signal_export_sub_dirs, "phase", "pedestrian_signal", signal_id,
                f"{self.year}-{self.month:02d}-{self.day:02d}.pkl"
            )

            # Load pedestrian phase profile data if it exists, otherwise extract it
            if os.path.exists(profile_filepath):
                df_pedestrian_phase_profile_id = pd.read_pickle(profile_filepath)
            else:
                df_pedestrian_phase_profile_id = self.extract_pedestrian_cycle_profile(signal_id=signal_id)

            # Define the file path for the vehicle cycle data of the specified signal and date
            profile_filepath = os.path.join(
                root_dir, relative_production_base_dir, *self.signal_export_sub_dirs, "cycle", "vehicle_signal", signal_id,
                f"{self.year}-{self.month:02d}-{self.day:02d}.pkl"
            )

            # Load vehicle cycle profile data if it exists, otherwise extract it
            if os.path.exists(profile_filepath):
                df_vehicle_cycle_profile_id = pd.read_pickle(profile_filepath)
            else:
                df_vehicle_cycle_profile_id = self.extract_vehicle_cycle_profile(signal_id=signal_id)


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

            # Revise signal export sub-directories
            signal_export_sub_dirs = self.signal_export_sub_dirs + ["cycle" + "pedestrian_signal", f"{signal_id}"]

            # Save the phase profile data as a pkl file in the export directory
            export_data(df=df_pedestrian_cycle_profile_id, 
                        base_dir=os.path.join(root_dir, relative_production_base_dir), 
                        filename=f"{self.year}-{self.month:02d}-{self.day:02d}", 
                        file_type="pkl", 
                        sub_dirs=signal_export_sub_dirs)

            return df_pedestrian_cycle_profile_id

        except Exception as e:
            # Log and raise an exception if any error occurs during processing
            logging.error(f"Error extracting pedestrian cycle profile for signal ID {signal_id}: {e}")
            raise CustomException(
                custom_message=f"Error extracting pedestrian cycle profile for signal ID {signal_id}: {e}",
                sys_module=sys
            )







