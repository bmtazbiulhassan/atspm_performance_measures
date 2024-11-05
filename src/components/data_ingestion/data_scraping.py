import pandas as pd
import time
import yaml
import sys
import os

from datetime import datetime

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from src.exception import CustomException
from src.logger import logging
from src.config import DataIngestionDirpath, get_relative_base_dirpath
from src.utils import get_root_directory, init_driver, export_data
from dotenv import load_dotenv


# Load the environment variables from the .env file
load_dotenv()

# Get the root directory of the project
root_dir = get_root_directory()


# Load the YAML configuration file (using absolute path)
with open(os.path.join(root_dir, "config/components", "data_ingestion.yaml"), "r") as file:
    config = yaml.safe_load(file)

# Retrieve settings for data scraping
config = config["data_scraping"]

# Retrieve wait time (in sec) to load web page while scraping
wait_time = config["wait_time"]

# Retrieve list of supported web browsers for scraping data
browser_types = config["browser_types"]


# Instantiate class to get directory paths
data_ingestion_dirpath = DataIngestionDirpath()

# Get relative base directory path for raw data
relative_raw_database_dirpath, _, _ = get_relative_base_dirpath()


def scrape_noemi_report(signal_id: str, siia_id: str):
    """
    Scrapes NOEMI report data for a specific SIIA ID and saves extracted tables as CSV files.

    Parameters:
    -----------
    signal_id : str
        Unique identifier for the signal, used to name the CSV files.
    siia_id : str
        SIIA ID used to access the specific report URL.

    Returns:
    --------
    None
        Saves extracted tables (from NOEMI report) to CSV files within the specified directory.
    """
    driver = None
    # Initialize a browser driver in sequence until successful
    for browser_type in browser_types:
        driver = init_driver(browser_type=browser_type)
        if driver:
            logging.info(f"{browser_type.capitalize()} driver initialized successfully.")
            break  # Proceed if a driver is initialized
    
    try:
        logging.info(f"Starting NOEMI report scraping for SIIA ID: {siia_id}")

        # Generate the URL by replacing SIIA ID placeholder in the base URL
        url = config["noemi"]["url"].replace("{}", str(siia_id))

        # Load configuration settings for tables and export sub-directories
        table_ids = config["noemi"]["table_ids"]

        # Open the report URL and wait for page to load
        driver.get(url)
        time.sleep(wait_time)

        # Parse page source with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract all tables from page source
        page_table_content = soup.find_all('table')
        for table_content in page_table_content:
            # Get ID of each table and check if it's in the specified table IDs
            table_id = table_content.get('id', None)
            if table_id and table_id in table_ids:
                # Extract column headers and table rows
                headers = [header.text.strip() for header in table_content.find('thead').find_all('th')]
                report_id = [[cell.text.strip() for cell in row.find_all('td')] for row in table_content.find('tbody').find_all('tr')]

                # Create DataFrame with extracted headers
                df_report_id = pd.DataFrame(data=report_id, columns=headers)

                logging.info(f"Extracted data for table '{table_id}' from NOEMI report for SIIA ID: {siia_id}")

                # Path (from database directory) to directory where scraped tables from NOEMI reports will be exported
                raw_report_dirpath, _ = data_ingestion_dirpath.get_data_scraping_dirpath(table_id=table_id)

                # Export as CSV file
                export_data(df=df_report_id, 
                            base_dirpath=os.path.join(root_dir, relative_raw_database_dirpath), 
                            sub_dirpath=raw_report_dirpath,
                            filename=f"{signal_id}", 
                            file_type="csv")

                report_filepath = os.path.join(root_dir, relative_raw_database_dirpath, raw_report_dirpath, 
                                               f"{signal_id}.csv")

                # Log the file saving operation and its path
                logging.info(f"Saved NOEMI report data to {report_filepath}")

    except Exception as e:
        # Log and raise exception if scraping fails
        logging.error(f"Error while scraping NOEMI report for SIIA ID: {siia_id} - {e}")
        raise CustomException(custom_message=str(e), sys_module=sys)
    
    finally:
        # Ensure the driver is closed after the process is complete
        if driver:
            driver.quit()


def scrape_event_data(day: int, month: int, year: int):
    """
    Scrapes ATSPM event data for a specific date from the Sunstore portal.
    
    Parameters:
    -----------
    day : int
        Day of the date (1-31).
    month : int
        Month of the date (1-12).
    year : int
        Year of the date (e.g., 2024).
        
    Returns:
    --------
    None
        Initiates download of data files for the specified date from the Sunstore portal.
        
    Raises:
    -------
    CustomException
        If any error occurs during login, navigation, or data extraction process.
    """
    # Convert day, month, and year to a pandas Timestamp for date comparison
    date = pd.Timestamp(datetime(year, month, day))
    date = date - pd.Timedelta(days=1)

    # Get today's date as a pandas Timestamp
    today_date = pd.Timestamp(datetime.today().date())
    date_range_start = today_date - pd.Timedelta(days=21)
    
    # Check if the requested date is within the allowable 20-day range and strictly before today
    if date < date_range_start or date >= today_date:
        raise CustomException(
            custom_message=(
                f"You can only scrape data between {date_range_start.strftime('%Y-%m-%d')} and {(today_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"
            ),
            sys_module=sys
        )

    # Path (from database directory) to directory where ATSPM event data scraped from Sunstore will be exported
    _, raw_event_dirpath = data_ingestion_dirpath.get_data_scraping_dirpath(month=month, year=year)

    # Absolute directory path to export event data
    event_dirpath = os.path.join(root_dir, relative_raw_database_dirpath, raw_event_dirpath)

    driver = None
    # Initialize a browser driver for each browser type in sequence until successful
    for browser_type in browser_types:
        driver = init_driver(browser_type=browser_type, download_dirpath=event_dirpath)
        if driver:
            logging.info(f"{browser_type.capitalize()} driver initialized successfully.")
            break  # Continue with scraping if a driver is successfully initialized
    
    try: 
        logging.info(f"Scraping ATSPM event data for date: {date}")

        # Retrieve the Sunstore URL from configuration and open it in the browser
        url = config["sunstore"]["url"]
        driver.get(url) 

        # Wait until the login link is clickable, then click to proceed to login
        login_link = WebDriverWait(driver, wait_time).until(
            EC.element_to_be_clickable((By.XPATH, "//a[@href='/login']"))
        )
        login_link.click()

        # Retrieve maximum login attempts allowed from configuration
        max_attempts = config["sunstore"]["max_attempts"]

        # Attempt login up to the maximum allowed attempts
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"\nAttempt {attempt} of {max_attempts} to Login")

                # Wait for input fields for username and password to be visible
                usermail = WebDriverWait(driver, wait_time).until(
                    EC.visibility_of_element_located((By.XPATH, "//input[@placeholder='Username or Email']"))
                )
                password = WebDriverWait(driver, wait_time).until(
                    EC.visibility_of_element_located((By.XPATH, "//input[@placeholder='Password']"))
                )

                # Input credentials (ideally, these should be secured and not hard-coded)
                usermail.send_keys(os.getenv("sunstore_usermail"))
                password.send_keys(os.getenv("sunstore_password"))

                # Click the login button to attempt logging in
                login_button = WebDriverWait(driver, wait_time).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@title='Log In']"))
                )
                login_button.click()

                # Verify login by locating the staged files link and clicking it to continue
                staged_files_button = WebDriverWait(driver, wait_time).until(
                    EC.element_to_be_clickable((By.XPATH, "//a[@href='/stagedfiles']"))
                )
                
                print("Login Successful!")
                staged_files_button.click()  # Navigate to the staged files section
                break  # Exit login loop after a successful login

            except Exception as e:
                # Log and inform the user of failed login attempts
                logging.warning(f"Login attempt {attempt} failed: {e}")
                print("Incorrect credentials. Please try again.")
                usermail.clear()
                password.clear()
                if attempt == max_attempts:
                    raise CustomException(custom_message="Maximum login attempts exceeded. Login failed.", sys_module=sys)

        # Wait for the staged files page to fully load
        time.sleep(wait_time)

        # Retrieve and parse the page source after navigating to staged files
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        page_table_content = soup.find_all('table')

        # Iterate over tables on the page to find the correct date and download link
        for table_content in page_table_content:
            # Extract column headers from the table to identify necessary columns
            table_headers = table_content.find('thead').find_all('th')
            headers = [header.text.strip() for header in table_headers]  # Column names

            # Retrieve column names for date and download link from configuration
            date_column_name = config["sunstore"]["date_column_name"]
            link_column_name = config["sunstore"]["link_column_name"]

            # Determine column indices for date and download link columns
            date_column_index = headers.index(date_column_name)
            link_column_index = headers.index(link_column_name)

            is_csv = False
            # Loop through table rows to find the row matching the specified date
            table_body = table_content.find('tbody').find_all('tr')
            for row in table_body:
                # Extract cell data for each row
                row_data = [cell.text.strip() for cell in row.find_all('td')]
                if date == pd.Timestamp(row_data[date_column_index]):

                    # Check if the row contains a CSV file
                    if "csv" in row_data:
                        is_csv = True

                    # Find the download link cell in the specified column and extract the link
                    link_cell = row.find_all("td")[link_column_index]
                    download_link_tag = link_cell.find("a", href=True)
                    
                    if download_link_tag:
                        # Retrieve and click the download link
                        download_link = download_link_tag["href"]
                        WebDriverWait(driver, wait_time).until(
                            EC.element_to_be_clickable((By.XPATH, f"//a[@href='{download_link}']"))
                        ).click()

                        logging.info(f"Clicked on download link: {download_link}")

                        time.sleep(wait_time)
                        
                        # Wait for the download to complete
                        wait_for_download_completion(download_dirpath=event_dirpath)  

                        logging.info("Download completed")
                        break  # Exit loop after clicking the download link

            # Break the outer loop if a CSV download link is found
            if is_csv:
                break

    except Exception as e:
        # Log and raise an exception if any error occurs during scraping
        logging.error(f"Error: '{e}' occurred while scraping ATSPM event data for date: {date}")
        raise CustomException(custom_message=str(e), sys_module=sys) 

    finally:
        # Ensure the driver is closed after completion
        if driver:
            driver.quit()


def wait_for_download_completion(download_dirpath: str):
    """
    Waits until all downloads in the specified directory are complete by checking for any files with the `.crdownload` extension.
    
    Parameters:
    -----------
    download_dirpath : str
        Path to directory where files are being downloaded.
        
    Returns:
    --------
    None
    """
    # Continuously check for incomplete downloads in the specified directory
    while any(filename.endswith('.crdownload') for filename in os.listdir(download_dirpath)):
        time.sleep(wait_time)  # Check every second if downloads have completed





