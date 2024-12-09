import os
import yaml

from src.utils import get_root_directory


# Get the root directory
root_dirpath = get_root_directory()

# Load the YAML configuration file (using absolute path)
with open(os.path.join(root_dirpath, "config/general_settings.yaml"), "r") as file:
    config = yaml.safe_load(file)


def get_relative_base_dirpath():
    """
    Retrieves the base directory paths for raw, interim, and production data.

    Returns:
    --------
    tuple
        A tuple containing base paths for raw, interim, and production data as specified in the configuration.
    """
    config_paths = config["relative_database_dirpath"]
    
    # Return paths for raw, interim, and production directories
    return config_paths["raw"], config_paths["interim"], config_paths["production"]


class DataIngestionDirpath:
    def __init__(self, district: str = config["district"]):
        """
        Initializes the DataIngestionDirpath class with the specified district.

        Parameters:
        -----------
        district : str, optional
            The district identifier (e.g., "fdot_d5" or "fdot_d7") to include in directory paths. Default is from `config["district"]`.
        """
        self.district = district

    def get_data_scraping_dirpath(self, table_id: str = "", month: int = None, year: int = None):
        """
        Constructs directory paths for data scraping tasks, including NOEMI report tables and ATSPM event data.

        Parameters:
        -----------
        table_id : str, optional
            Identifier for the table within the NOEMI reports. Default is an empty string.
        month : int, optional
            The month for which ATSPM event data is stored. Default is None.
        year : int, optional
            The year for which ATSPM event data is stored. Default is None.

        Returns:
        --------
        tuple
            A tuple containing paths to the NOEMI report directory and ATSPM event data directory.
        """
        # Directory path within the base directory to store tables from NOEMI reports, formatted by district and table ID
        raw_report_dirpath = config["raw_report_dirpath"].format(
            district=self.district, 
            table_id=table_id
        )

        if month is not None:
            month = f"{month:02d}"
            
        
        # Directory path within the base directory to store ATSPM event data, formatted by district, year, and month
        raw_event_dirpath = config["raw_event_dirpath"].format(
            district=self.district, 
            year=year, 
            month=month
        )
        
        return raw_report_dirpath, raw_event_dirpath

    def get_data_sorting_dirpath(self, signal_id: str = "", month: int = None, year: int = None):
        """
        Constructs directory paths for storing sorted ATSPM event data.

        Parameters:
        -----------
        signal_id : str, optional
            Unique identifier for the signal, used to create specific directories. Default is an empty string.
        month : int, optional
            The month for which the event data is stored. Default is None.
        year : int, optional
            The year for which the event data is stored. Default is None.

        Returns:
        --------
        tuple
            A tuple containing the base directory path where event data is stored and the interim directory path 
            for storing sorted event data based on the specified district, year, month, and signal ID.
        """
        if month is not None:
            month = f"{month:02d}"
            
        # Directory path within the base directory where ATSPM event data, formatted by district, year, and month, is stored.
        raw_event_dirpath = config["raw_event_dirpath"].format(
            district=self.district, 
            year=year, 
            month=month
        )

        # Directory path within the base directory to store sorted ATSPM event data, formatted by district, and signal ID
        interim_event_dirpath = config["interim_event_dirpath"].format(
            district=self.district, 
            signal_id=signal_id
        )

        return raw_event_dirpath, interim_event_dirpath
    
    def get_data_preprocessing_dirpath(self, table_id: str = ""):
        """
        Constructs directory paths for preprocessing data from NOEMI reports and storing signal configuration data.

        Parameters:
        -----------
        table_id : str, optional
            Identifier for the table within the NOEMI reports. Default is an empty string.

        Returns:
        --------
        tuple
            A tuple containing paths for the NOEMI report directory and the interim configuration directory.
        """
        
        # Directory path within the base directory where tables from NOEMI reports, formatted by district and table ID, are stored.
        raw_report_dirpath = config["raw_report_dirpath"].format(
            district=self.district, 
            table_id=table_id
        )

        # Directory path within the base directory to store signal configuration ("intersection" + "lane" tables) data.
        interim_config_dirpath = config["interim_config_dirpath"].format(
            district=self.district
        )

        return raw_report_dirpath, interim_config_dirpath


class FeatureExtractionDirpath:
    def __init__(self, district: str = config["district"]):
        """
        Initializes the DataIngestionDirpath class with the specified district.

        Parameters:
        -----------
        district : str, optional
            The district identifier (e.g., "fdot_d5" or "fdot_d7") to include in directory paths. Default is from `config["district"]`.
        """
        self.district = district

    def get_data_quality_check_dirpath(self, signal_id: str = "", event_type: str = ""):
        """
        Constructs directory paths for data quality checks, including sorted ATSPM event data, 
        signal configuration data, and quality check results.

        Parameters:
        -----------
        signal_id : str, optional
            Unique identifier for the signal, used to create specific directories. Default is an empty string.
        event_type : str, optional
            Type of event data being checked (e.g., "vehicle_signal" or "vehicle_traffic"). Default is an empty string.

        Returns:
        --------
        tuple
            A tuple containing paths for the interim directory with sorted ATSPM event data, 
            interim directory with signal configuration data, and production directory for quality check results.
        """

        # Directory path within the base directory sorted ATSPM event data, formatted by district, and signal ID, is stored.
        interim_event_dirpath = config["interim_event_dirpath"].format(
            district=self.district, 
            signal_id=signal_id
        )

        # Directory path within the base directory where signal configuration ("intersection" + "lane" tables) data, is stored.
        interim_config_dirpath = config["interim_config_dirpath"].format(
            district=self.district
        )

        # Directory path within the base directory to store data quality check results.
        production_check_dirpath = config["production_check_dirpath"].format(
            district=self.district, 
            event_type=event_type
        )

        return interim_event_dirpath, interim_config_dirpath, production_check_dirpath

    def get_feature_extraction_dirpath(
            self, signal_id: str = "", resolution_level: str = "", event_type: str = "", feature_name: str = ""
        ):
        """
        Constructs directory paths for feature extraction tasks, including paths to sorted ATSPM event data,
        signal configuration data, signal profile data, and specific SPaT and traffic features.

        Parameters:
        -----------
        signal_id : str, optional
            Unique identifier for the signal, used to create specific directories. Default is an empty string.
        resolution_level : str, optional
            Specifies the level of resolution for feature extraction (e.g., "phase" or "cycle"). Default is an empty string.
        event_type : str, optional
            Type of event data used for feature extraction ("vehicle_signal", "pedestrian_signal", "vehicle_traffic", or "pedestrian_traffic"). 
            Default is an empty string.
        feature_name : str, optional
            Name of the specific feature for extraction (e.g., "volume", "occupancy", etc.). Default is an empty string.

        Returns:
        --------
        tuple
            A tuple containing paths for the interim directory with sorted ATSPM event data, 
            interim directory with signal configuration data, production directory for signal profiles,
            and production directory for SPaT and traffic features.
        """
        # Directory path within the base directory sorted ATSPM event data, formatted by district, and signal ID, is stored.
        interim_event_dirpath = config["interim_event_dirpath"].format(
            district=self.district, 
            signal_id=signal_id
        )

        # Directory path within the base directory where signal configuration ("intersection" + "lane" tables) data, is stored.
        interim_config_dirpath = config["interim_config_dirpath"].format(
            district=self.district
        )
        
        # Directory path within the base directory to store vehicle and pedestrian signal profile data.
        production_signal_dirpath = config["production_signal_dirpath"].format(
            district=self.district,
            resolution_level=resolution_level,
            event_type=event_type,
            signal_id=signal_id
        )

        # Directory path within the base directory to store vehicle and pedestrian-related signal and traffic features.
        production_feature_dirpath = config["production_feature_dirpath"].format(
            district=self.district,
            resolution_level=resolution_level,
            event_type=event_type,
            feature_name=feature_name,
            signal_id=signal_id
        )
        
        return interim_event_dirpath, interim_config_dirpath, production_signal_dirpath, production_feature_dirpath


