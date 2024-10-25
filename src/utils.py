import sys
import os

from pathlib import Path

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.chrome.service import Service

from src.exception import CustomException
from src.logger import logging


def get_root_directory():
    """
    Determines the root directory of the project by searching for any of the following files:
    'config.yaml', 'setup.py', 'requirements.txt', 'readme.md', 'pyproject.toml'.
    
    Returns:
    --------
    Path
        The path to the root directory containing one of the specified marker files.
    
    Raises:
    -------
    CustomException
        If none of the marker files are found in the root directory.
    """
    # List of marker files to identify the root directory
    marker_files = ["config.yaml", "setup.py", "requirements.txt", "readme.md", "pyproject.toml"]

    # Start from the directory where this function is called
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = Path(__file__).parent
    
    # Traverse upwards through parent directories
    for parent_dir in current_dir.parents:
        if any(os.path.exists(parent_dir / marker_file) for marker_file in marker_files):
            return parent_dir  # Return the directory as root if any marker file is found

    # Log the error if if no marker file is found in the root directory and raise a custom exception
    error_message = f"Root directory not found: No marker file in {marker_files} detected in the root directory."
    logging.error(f"{error_message}")
    raise CustomException(custom_message=error_message, sys_module=sys)

def init_driver(browser_type: str = "chrome"):
    """
    Initializes a web driver based on the browser type.

    Parameters:
    -----------
    browser_type: str
        Type of the browser ("chrome", "firefox", "edge").

    Returns:
    --------
    driver: WebDriver
        The initialized web driver.
    """
    try:
        # Log the browser initialization process
        logging.info(f"Initializing browser: {browser_type}")

        if browser_type == "chrome":
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        elif browser_type == "firefox":
            driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
        elif browser_type == "edge":
            driver = webdriver.Edge(EdgeChromiumDriverManager().install())
        else: # Raise an exception if an unsupported browser is specified
            logging.info(f"Unsupported browser: {browser_type}")
            raise CustomException(custom_message=f"Unsupported browser", sys_module=sys)
        
        return driver
    
    except Exception as e:
        # Log the error if initialization fails and raise a custom exception
        logging.info(f"Error initializing {browser_type} driver: {e}")
        raise CustomException(custom_message=e, sys_module=sys)
    


