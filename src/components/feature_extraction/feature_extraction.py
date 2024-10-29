import pandas as pd
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import get_root_directory, get_column_name_by_partial_name


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
            start_date = pd.to_datetime(f'{year}-{month:02d}-{day:02d} 00:00:00')
            end_date = pd.to_datetime(f'{year}-{month:02d}-{day:02d} 23:59:59')
            
            # Convert the timestamp column to datetime if it is not already
            if not pd.api.types.is_datetime64_any_dtype(df_event[dict_column_names["time"]]):
                df_event[dict_column_names["time"]] = pd.to_datetime(df_event[dict_column_names["time"]])
            
            # Filter the DataFrame for rows within the specified date range
            df_event = df_event[((df_event[dict_column_names["time"]] >= start_date) &
                                 (df_event[dict_column_names["time"]] <= end_date))]
            
            return df_event
        
        except Exception as e:
            raise CustomException(custom_message=f"Error filtering by day: {e}", sys_module=sys)







