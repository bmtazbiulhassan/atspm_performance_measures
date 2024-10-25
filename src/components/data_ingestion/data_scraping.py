import pandas as pd
import time
import yaml
import sys
import os

from bs4 import BeautifulSoup

from src.exception import CustomException
from src.logger import logging
from src.utils import get_root_directory, init_driver


# Get the root directory
root_dir = get_root_directory()

def scrape_noemi_report(signal_id: str, siia_id: str):
    """
    Scrapes NOEMI report data from a specified URL for a given SIIA ID and saves the extracted tables as CSV files.

    Parameters:
    -----------
    signal_id : str
        Identifier for the signal, used to name the CSV files.
    siia_id : str
        SIIA ID used to access the specific report URL.

    Returns:
    --------
    None
        Saves the extracted tables to CSV files within the specified directory.
    """
    try: 
        # Log the start of the scraping process for the given SIIA ID
        logging.info(f"Scraping NOEMI reports for SIIA ID: {siia_id}")

        # Load the YAML configuration file (using absolute path)
        with open(os.path.join(root_dir, 'config.yaml'), "r") as file:
            config = yaml.safe_load(file)

        # Retrieve the URL template from the YAML config and replace '{}' with the actual SIIA ID
        url = config["noemi_report"]["url"].replace("{}", f"{siia_id}")

        # Retrieve the table IDs to be extracted from the YAML config
        table_ids = config["noemi_report"]["table_ids"]

        # Retrieve the parent directory path for saving tables from the YAML config
        parent_dir = config["noemi_report"]["parent_dir"]

        # Initialize the web driver for Chrome
        driver = init_driver("chrome")

        # Open the specified report URL in the web driver
        driver.get(url) 
        time.sleep(5)  # Wait (5 sec) for the page to load completely

        # Parse the page source using BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract all tables from the page source
        report_content = soup.find_all('table')
        for i in range(len(report_content)):
            # Retrieve the ID of each table
            table_id = report_content[i]['id']

            # Continue only if the table ID is in the specified table IDs to extract
            if table_id not in table_ids:
                continue

            # Select the table based on the table ID
            table = report_content[i]

            # Extract column names from the table header
            column_names = [table_content.text for table_content in table.find('thead').find_all('th')]

            # Extract table content from the table body rows
            table_content = table.find('tbody').find_all('tr')

            data = []
            for j in range(len(table_content)):
                # Extract text content for each cell in the row
                data.append([content.text for content in table_content[j]])

            # Convert the extracted data into a DataFrame with the column names
            df_data = pd.DataFrame(data=data, columns=column_names) 

            # Log the success of the scraping process for the current table
            logging.info(f"Successfully extracted data for table {table_id} from the NOEMI report for SIIA ID: {siia_id}")

            # Define the path for saving the CSV file in the respective table's folder
            table_path = os.path.join(root_dir, parent_dir, table_id)
            os.makedirs(table_path, exist_ok=True)  # Create directory if it does not exist

            # Save the DataFrame as a CSV file named by the signal_id within the table's folder
            df_data.to_csv(f"{table_path}/{signal_id}.csv", index=False)

            # Log the file save operation with the path
            logging.info(f"Saving scraped NOEMI report to...{table_path}/{signal_id}.csv")

    except Exception as e:
        # Log the error if scraping fails and raise a custom exception
        logging.error(f"Error: '{e}' raised while scraping NOEMI reports for SIIA ID: {siia_id}")
        raise CustomException(custom_message=e, sys_module=sys)








