import sys

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.chrome.service import Service

from src.exception import CustomException
from src.logger import logging


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
    


