import pandas as pd
import yaml
import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import get_root_directory, get_column_name_by_partial_name, float_to_int


# Get the root directory of the project
root_dir = get_root_directory()

# Load the YAML configuration file (using absolute path)
with open(os.path.join(root_dir, "config.yaml"), "r") as file:
    config = yaml.safe_load(file)

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
            Sequence of event codes to filter (e.g., [1, 8, 10, 11] or [82, 81]).

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
    def __init__(self, relative_event_import_parent_dir: str, relative_config_import_dir: str, relative_signal_export_parent_dir: str, day: int, month: int, year: int):
        """
        Initializes the TrafficSignalProfile class with necessary paths.

        Parameters:
        -----------
        relative_event_import_parent_dir : str
            Relative path to directory which contains sub-directories where event data for signals is stored.
        relative_config_import_dir : str
            Relative path to directory where configuration data for signals is stored.
        relative_signal_export_parent_dir : str
            Relative path to directory which contains sub-directories (which further contains sub-directors) 
            where extracted signal profile for phase and cycle will be saved.
        day : int
            Day of the desired date (1-31).
        month : int
            Month of the desired date (1-12).
        year : int
            Year of the desired date (e.g., 2024).
        """
        self.relative_event_import_parent_dir = relative_event_import_parent_dir
        self.relative_config_import_dir = relative_config_import_dir
        self.relative_signal_export_parent_dir = relative_signal_export_parent_dir

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
        # Load configuration data for the specific signal
        config_filepath = os.path.join(root_dir, self.relative_config_import_dir, f"{signal_id}.csv")
        df_config_id = pd.read_csv(config_filepath)

        # Convert columns with float values to integer type where applicable
        df_config_id = float_to_int(df_config_id)

        # Dynamically fetch the phase column name based on partial match
        dict_column_names = {"phase": get_column_name_by_partial_name(df=df_config_id, partial_name="phase")}
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

    def extract_phase_profile(self, signal_id: str, signal_type: str):
        """
        Load, prepare, and extract phase data for a specific signal (vehicle or pedestrian).

        Parameters:
        -----------
        signal_id : str
            Unique identifier for the signal.
        signal_type : str
            Type of signal to process - "vehicle" or "pedestrian".

        Returns:
        --------
        pd.DataFrame
            DataFrame containing extracted phase data.
        """
        # Load file path for event data
        event_filepath = os.path.join(
            root_dir, self.relative_event_import_parent_dir, signal_id, f"{self.year}-{self.month:02d}-{self.day:02d}.csv"
            )

        # Set event sequence and mapping based on signal type
        if signal_type == "vehicle":
            # Mapping event codes to respective phase times
            event_code_map = {
                "greenBegin": 1, "greenEnd": 8,
                "yellowBegin": 8, "yellowEnd": 10,
                "redClearanceBegin": 10, "redClearanceEnd": 11
            }
            valid_event_sequence = config["sunstore"]["valid_event_sequence"]["vehicle_signal"]

        elif signal_type == "pedestrian":
            # Mapping event codes for pedestrian phases
            event_code_map = {
                "pedestrianWalkBegin": 21, "pedestrianWalkEnd": 22,
                "pedestrianClearanceBegin": 22, "pedestrianClearanceEnd": 23,
                "pedestrianDontWalkBegin": 23
            }
            valid_event_sequence = config["sunstore"]["valid_event_sequence"]["pedestrian_signal"]

        else:
            logging.error(f"{signal_type} is not valid.")
            raise CustomException(
                custom_message=f"{signal_type} is not valid. Must be 'vehicle' or 'pedestrian'.", 
                sys_module=sys
            )

        try:
            # Load and prepare configuration data with barrier numbers if applicable
            df_config_id = self.add_barrier_no(signal_id=signal_id)

            # Load event data and prepare column names dynamically
            df_event_id = pd.read_pickle(event_filepath)
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
            phase_profile = []
            for phase_no in sorted(df_event_id[dict_column_names["param"]].unique()):
                # Filter data for each phase number and assign sequence IDs
                df_event_phase = df_event_id[df_event_id[dict_column_names["param"]] == phase_no]
                df_event_phase = self.add_event_sequence_id(df_event_phase, valid_event_sequence=valid_event_sequence)

                for sequence_id in df_event_phase["sequenceID"].unique():
                    df_event_sequence = df_event_phase[df_event_phase["sequenceID"] == sequence_id]
                    current_event_sequence = df_event_sequence[dict_column_names["code"]].tolist()
                    correct_flag = int(current_event_sequence == valid_event_sequence)

                    # Generate phase information dictionary, dynamically adding timestamps for each event
                    phase_info = {
                        "signalID": signal_id,
                        "phaseNo": phase_no,
                        "correctSequenceFlag": correct_flag,
                        **{key: df_event_sequence[df_event_sequence[dict_column_names["code"]] == code][dict_column_names["time"]].iloc[0]
                           if code in current_event_sequence else pd.NaT for key, code in event_code_map.items()}
                    }
                    # Conditionally add "barrierNo" if the signal type is "vehicle"
                    if signal_type == "vehicle":
                        phase_info["barrierNo"] = config["noemi"]["barrier_map"].get(int(phase_no), 0)

                    phase_profile.append(phase_info)

            # Convert phase profile to DataFrame and create a pseudo timestamp for sorting
            df_phase_profile = pd.DataFrame(phase_profile)
            time_columns = [column for column in df_phase_profile.columns if column.endswith("Begin") or column.endswith("End")]
            df_phase_profile["pseudoTimestamp"] = df_phase_profile[time_columns].bfill(axis=1).iloc[:, 0]
            df_phase_profile = df_phase_profile.sort_values(by="pseudoTimestamp").reset_index(drop=True)
            df_phase_profile.drop(columns=["pseudoTimestamp"], inplace=True)

            # Export phase data to appropriate directory based on signal type
            absolute_signal_export_dir = os.path.join(
                root_dir, self.relative_signal_export_parent_dir, signal_type, signal_id,
                )
            os.makedirs(absolute_signal_export_dir, exist_ok=True)
            df_phase_profile.to_csv(f"{absolute_signal_export_dir}/{self.year}-{self.month:02d}-{self.day:02d}.csv", 
                                    index=False)

            return df_phase_profile

        except Exception as e:
            logging.error(f"Error extracting phase profile for signal ID {signal_id}: {e}")
            raise CustomException(
                custom_message=f"Error extracting phase profile for signal ID {signal_id}: {e}", 
                sys_module=sys
                )

    def extract_vehicle_phase(self, signal_id: str):
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
        return self.extract_phase_profile(signal_id, signal_type="vehicle")

    def extract_pedestrian_phase(self, signal_id: str):
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
        return self.extract_phase_profile(signal_id, signal_type="pedestrian")











