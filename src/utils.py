import os
import re
import sys
import pandas as pd

from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.service import Service

from src.exception import CustomException


def get_root_directory(marker: str = 'src'):
    """
    Determines the root directory of the project by searching for a specified marker (file or directory) inside the root directory.

    Parameters:
    -----------
    marker : str, optional
        The marker is a file or directory name used to identify the project root directory. Default is 'src'.

    Returns:
    --------
    Path
        The path to the root directory containing the specified marker file or directory.

    Raises:
    -------
    CustomException
        If the marker file or directory is not found in any parent directory.
    """
    # Start from the directory where this function is defined
    current_dir = Path(__file__).parent

    # Traverse upwards through parent directories
    for parent_dir in current_dir.parents:
        if (parent_dir / marker).exists():
            return parent_dir  # Return the directory as root if marker file is found

    # Raise an error if the marker file is not found
    raise CustomException(
        custom_message=f"Root directory could not be determined. The specified marker '{marker}' was not found.",
        sys_module=sys
    )


def init_driver(browser_type: str = "chrome", download_dirpath: str = None):
    """
    Initializes a web driver based on the browser type and configures download settings.
    
    Parameters:
    -----------
    browser_type : str
        Type of the browser. Currently, only "chrome" is supported.
    download_dirpath : str, optional
        Path to the directory where downloaded files should be saved. If not specified, the browser’s default download path is used.
        
    Returns:
    --------
    driver : WebDriver
        The initialized Chrome web driver instance configured for downloads.
        
    Raises:
    -------
    CustomException
        If an unsupported browser type is specified.
    """
    if browser_type == "chrome":
        # Configure Chrome options for customized settings
        chrome_options = webdriver.ChromeOptions()

        # Set Chrome preferences for automatic download if a custom download directory is provided
        if download_dirpath:
            prefs = {
                "download.default_directory": download_dirpath,  # Path where files will be downloaded
                "download.prompt_for_download": False,       # Disable download prompts
                "safebrowsing.enabled": True                 # Enable safe browsing for secure downloads
            }
            chrome_options.add_experimental_option("prefs", prefs)

        # Initialize Chrome driver with the specified options
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)

        # Return the initialized Chrome driver
        return driver
    else:
        # Raise a custom exception if an unsupported browser type is specified
        raise CustomException(
            custom_message=f"Unsupported browser: {browser_type.capitalize()}. Currently, only 'chrome' is supported.",
            sys_module=sys
        )


def get_column_name_by_partial_name(df: pd.DataFrame, partial_name: str):
    """
    Finds and returns the first column name in the DataFrame that matches the given partial name.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to search for the column.
    partial_name : str
        The partial name to search for in the column names.

    Returns:
    --------
    str
        The name of the first column that matches the partial name.
    
    Raises:
    -------
    CustomException
        If no column with the given partial name is found.
    """
    # Compile a regular expression pattern for case-insensitive search
    pattern = re.compile(partial_name, re.IGNORECASE)

    # Search for a matching column
    matching_column_name = next((col for col in df.columns if pattern.search(col)), None)
    
    # Raise an error if no column is found
    if matching_column_name is None:
        raise CustomException(f"No column with partial name '{partial_name}' found.", sys_module=sys)
    
    return matching_column_name


def float_to_int(df: pd.DataFrame):
    """
    Convert float64 columns to Int64 in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame in which to convert columns.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with float64 columns converted to Int64.
    """
    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Check if the column's data type is float64
        if df[column].dtype == 'float64':
            df = df.copy()
            
            # Check if the column contains any NaN values
            if df[column].isna().any():
                # Convert the column to nullable integer (Int64)
                df[column] = df[column].astype('Int64')
            else:
                # Convert the column to regular integer (int)
                df[column] = df[column].astype(int)
        
    return df


def get_single_unique_value(df: pd.DataFrame, column_name: str):
    """
    Fetches the unique value of a specified column in a DataFrame if there is only one unique value.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data.
    column_name : str
        The name of the column to check for a single unique value.
        
    Returns:
    --------
    The unique value of the column if there is only one unique value, otherwise None.
    
    Raises:
    -------
    KeyError
        If the specified column does not exist in the DataFrame.
    """
    # Check if the specified column exists in the DataFrame
    if column_name not in df.columns:
        raise CustomException(
            custom_message=f"Column '{column_name}' does not exist in the DataFrame.", sys_module=sys
            )
    
    # Get the unique values in the specified column
    unique_values = df[column_name].unique()
    
    try:
        # Check if there is only one unique value in the column
        if len(unique_values) == 1:
            return unique_values[0]
    except:
        raise CustomException(
            custom_message=f"Column '{column_name}' has more than one unique value.", sys_module=sys
            )


def create_dict(int_keys=None, str_keys=None, list_keys=None, dict_keys=None):
    """
    Creates a dictionary with specified keys and types of default values.

    This function creates a dictionary where specified keys are initialized with default values based on their type:
    - Integer keys are initialized to 0.
    - List keys are initialized to empty lists.
    - String keys are initialized to empty strings.
    - Dictionary keys are initialized to empty dictionaries.

    Parameters:
    -----------
    int_keys : list, optional
        List of keys to be initialized with integer values (0). Defaults to an empty list if None.
    str_keys : list, optional
        List of keys to be initialized with empty string values (''). Defaults to an empty list if None.
    list_keys : list, optional
        List of keys to be initialized with empty list values ([]). Defaults to an empty list if None.
    dict_keys : list, optional
        List of keys to be initialized with empty dictionary values ({}). Defaults to an empty list if None.

    Returns:
    --------
    dict
        A dictionary with initialized keys and default values based on the specified types.

    Example:
    --------
    >>> init_dict(int_keys=['count'], list_keys=['items'], str_keys=['name'], dict_keys=['info'])
    {'count': 0, 'items': [], 'name': '', 'info': {}}
    """
    # Set default empty lists if parameters are None
    int_keys = int_keys or []
    str_keys = str_keys or []
    list_keys = list_keys or []
    dict_keys = dict_keys or []

    # Creates the dictionary with specified types
    default_dict = {key: 0 for key in int_keys}          # Integer keys with value 0
    default_dict.update({key: "" for key in str_keys})   # String keys with empty strings
    default_dict.update({key: [] for key in list_keys})  # List keys with empty lists
    default_dict.update({key: {} for key in dict_keys})  # Dictionary keys with empty dictionaries

    return default_dict


def load_data(base_dirpath: str, sub_dirpath: str, filename: str, file_type: str = "csv"):
    """
    Loads a DataFrame from a specified directory and file type.

    Parameters:
    -----------
    base_dirpath : str
        The base directory path within the root directory where the data is stored.
    sub_dirpath : str
        The relative path within the base directory for navigating to the file.
    filename : str
        The name of the file (without extension) to load.
    file_type : str, optional
        The type of file to load ("csv" or "pkl" for pickle). Default is "csv".

    Returns:
    --------
    pd.DataFrame
        The loaded DataFrame.

    Raises:
    -------
    CustomException
        If the specified file does not exist or an unsupported file type is provided.
    """
    try:
        # Combine ROOT_DIR with base_dirpath and sub_dirpath to form the full file path
        file_dirpath = os.path.join(get_root_directory(), base_dirpath, sub_dirpath)
        filepath = os.path.join(file_dirpath, f"{filename}.{file_type}")

        # Check if the file exists
        if not os.path.exists(filepath):
            raise CustomException(custom_message=f"File {filepath} not found", sys_module=sys)

        # Load the file based on the file type
        if file_type == "csv":
            df = pd.read_csv(filepath)
        elif file_type == "pkl":
            df = pd.read_pickle(filepath)
        else:
            raise CustomException(custom_message=f"Unsupported file type: {file_type}. Choose 'csv' or 'pkl'.", 
                                  sys_module=sys)
        
        return df

    except Exception as e:
        raise CustomException(custom_message=f"Unexpected error while loading data: {str(e)}", 
                              sys_module=sys)


def export_data(df: pd.DataFrame, base_dirpath: str, sub_dirpath: str, filename: str, file_type: str = "csv"):
    """
    Exports a DataFrame to a specified directory and file type.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to export.
    base_dirpath : str
        The base directory path within the root directory for saving the data.
    sub_dirpath : str
        The relative path within the base directory where the file will be saved.
    filename : str
        The name of the file (without extension) for the exported data.
    file_type : str, optional
        The type of file to save ("csv" or "pkl" for pickle). Default is "csv".

    Raises:
    -------
    CustomException
        If an unsupported file type is provided or any error occurs during export.
    """
    try:
        # Construct the full file path by combining ROOT_DIR, base_dirpath, and sub_dirpath
        file_dirpath = os.path.join(get_root_directory(), base_dirpath, sub_dirpath)
        os.makedirs(file_dirpath, exist_ok=True)
        filepath = os.path.join(file_dirpath, f"{filename}.{file_type}")

        # Export the data based on the file type
        if file_type == "csv":
            df.to_csv(filepath, index=False)
        elif file_type == "pkl":
            df.to_pickle(filepath)
        else:
            raise CustomException(custom_message=f"Unsupported file type: {file_type}. Choose 'csv' or 'pkl'.", 
                                  sys_module=sys)

    except Exception as e:
        raise CustomException(custom_message=f"Unexpected error while exporting data: {str(e)}", 
                              sys_module=sys)






