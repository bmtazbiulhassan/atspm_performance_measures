import sys


def get_error_details(exception_message, sys_module: sys):
    """
    Retrieves detailed error information, including the file name,
    line number, and exception message where the error occurred.
    
    Parameters:
    -----------
    exception_message : str
        The message to display in the exception (can be custom or from the actual raised exception).
    sys_module : sys
        The sys module used to extract traceback information.

    Returns:
    --------
    str
        A formatted error message with the file name, line number, and exception message.
    """
    # Extracts the traceback object
    _, _, traceback_info = sys_module.exc_info()

    # If traceback_info is None, return a basic error message
    if traceback_info is None:
        return f"Error: {exception_message} - No traceback information available."

    # Get the file name where the exception occurred
    file_name = traceback_info.tb_frame.f_code.co_filename

    # Get the line number where the exception occurred
    line_number = traceback_info.tb_lineno

    # Generate a detailed error message with file name, line number, and exception description
    error_message = f"Error occurred in script [{file_name}] at line [{line_number}]: {exception_message}"
    
    return error_message


class CustomException(Exception):
    """
    A custom exception class that generates detailed error messages,
    including the script name, line number, and the original exception message.
    """
    def __init__(self, custom_message=None, sys_module: sys = None):
        """
        Initializes the CustomException with a detailed error message.

        Parameters:
        -----------
        custom_message : str, optional
            A custom error message provided by the user. If None, the original exception message is used.
            Original exception message can also be provided as custom error message.
        sys_module : sys, optional
            The sys module used to extract traceback information.
        """
        # If no custom message is provided, use the original exception's message
        if custom_message is None:
            # Retrieve the original exception message from sys
            custom_message = str(sys_module.exc_info()[1])  # Get the actual exception message

        # Call the base Exception class with the custom message (or actual exception message)
        super().__init__(custom_message)

        # Store the detailed error message using the format_error_details function
        self.error_message = get_error_details(custom_message, sys_module)

    def __str__(self):
        """
        Returns the detailed error message when the exception is printed.

        Returns:
        --------
        str
            The detailed error message, including file name and line number.
        """
        return self.error_message




