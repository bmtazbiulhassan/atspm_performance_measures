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


def init_driver(browser_type: str = "chrome", download_dir: str = None):
    """
    Initializes a web driver based on the browser type and configures download settings.
    
    Parameters:
    -----------
    browser_type : str
        Type of the browser. Currently, only "chrome" is supported.
    download_dir : str, optional
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
        if download_dir:
            prefs = {
                "download.default_directory": download_dir,  # Path where files will be downloaded
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
            # Convert the column to Int64 type
            df[column] = df[column].astype('Int64')
    
    return df









