from setuptools import setup, find_packages

def get_requirements(filepath: str) -> list[str]:
    """
    Reads the "requirements.txt" file, processes it to extract package dependencies, 
    and returns a list of requirements, with potential cleanup if needed.
    
    Parameters:
    -----------
    filepath : str
        The path to the "requirements.txt" file containing package dependencies.

    Outputs:
    --------
    list[str] : 
        A list of package requirements (dependencies) with any unnecessary characters 
        (like newlines) removed, and any development flags like "-e ." omitted.
    """
    
    # Initialize an empty list to hold the requirements.
    requirements = []

    # Open the file at the given filepath to read the contents.
    with open(filepath) as file:
        # Read all the lines from the file and store them in the requirements list.
        requirements = file.readlines()

        # Remove newline characters "\n" from each requirement to clean up the entries.
        requirements = [requirement.replace("\n", "") for requirement in requirements]

        # If the list contains "-e ." (which is used for editable installs), remove it.
        # This ensures that editable installs are not included in production dependencies.
        if "-e ." in requirements:
            requirements.remove("-e .")
    
    # Return the processed list of requirements.
    return requirements


setup(
    name="citysignal",
    version="1.0.0",
    author="B M Tazbiul Hassan Anik",
    author_email="anik.bmtazbiulhassan@gmail.com", 
    packages=find_packages(),
    install_requires=get_requirements(filepath="requirements.txt")
)





