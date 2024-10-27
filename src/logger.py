import os  
import logging 

from datetime import datetime  

from src.utils import get_root_directory


# Get the root directory of the project
root_dir = get_root_directory()

# Generate a log file name with the current date and time
# This helps in creating unique log files based on the time the script runs.
log_filename = f"{datetime.now().strftime('%m-%d-%Y@%H-%M-%S')}.log"

# Define the directory where logs will be saved
# 'logs' folder will be created inside the 'reports' folder in the current working directory (if not already present).
log_dir = os.path.join(root_dir, "logs")

# Create the logs directory if it doesn't exist
# The `exist_ok=True` ensures no error is raised if the folder already exists.
os.makedirs(log_dir, exist_ok=True)

# Create the full log file path by combining the logs directory and log file name
log_filepath = os.path.join(log_dir, log_filename)

# Configure the logging settings
logging.basicConfig(
    filename=log_filepath,  # Set the log file path
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Set the log format
    level=logging.INFO,  # Set the logging level (INFO level will capture all INFO, WARNING, ERROR, etc.)
)

