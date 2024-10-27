import sys

from pathlib import Path
from datetime import datetime

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

    
    


