import sys
sys.path.append("./")
import os
import logging
from colorlog import ColoredFormatter
from datetime import datetime
from Utils.Enums.logger_enums import LogLevel

class CustomLogger(logging.Logger):

    def __init__(self, name, dir=None):
        """Initialize the CustomLogger class.

        Args:
            name (str): The name of the logger.
        """
        super().__init__(name)  # Call the parent class's init function
        self.handlers = []  # Initialize an empty handlers list
        # Define the directory name
        if isinstance(dir, str):
            self.log_dir = os.path.join(dir, 'Log')
        else:
            self.log_dir = 'Log'

        # Set up a temporary logger for handling possible errors during logger configuration
        self.temp_logger = logging.getLogger(name + "_temp")
        self.temp_logger.addHandler(logging.StreamHandler())
        self.temp_logger.setLevel(logging.DEBUG)

        self.__configure()  # Configure the logger

    def __create_log_dir(self):
        """Create the log directory if it doesn't already exist.

        Returns:
            str: The name of the log directory.
        """
        try:
            if not os.path.exists(self.log_dir):  # If the directory does not exist,
                os.makedirs(self.log_dir)  # create it
        except Exception as e:
            # Log the error with the temporary logger
            self.temp_logger.error(f"Error while creating log directory: {e}")
        return self.log_dir

    def __configure(self):
        """Configure the logger's settings and handlers.
        """
        # Define the log format
        log_format = "%(log_color)s [%(asctime)s] [%(levelname)s] [%(name)s]  =>>>  %(reset)s %(message)s"
        formatter = ColoredFormatter(log_format, datefmt="%d-%b-%y %H:%M:%S")

        # Create a console handler and configure it
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        # Create the log directory
        log_dir = self.__create_log_dir()

        # Define the log file name
        log_file = os.path.join(
            log_dir, f'{datetime.now().strftime("%d-%m-%Y")}.log')

        try:
            # Create a file handler and configure it
            file_handler = logging.FileHandler(log_file)
        except Exception as e:
            # Log the error with the temporary logger
            self.temp_logger.error(f"Error while opening the log file: {e}")
            file_handler = None  # If an error occurred, set the file handler to None

        # If the file handler is successfully created, add it to the handlers list
        if file_handler is not None:
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.handlers = [console_handler, file_handler]

        # Add each handler in the handlers list to the logger
        for handler in self.handlers:
            self.addHandler(handler)

        # Set the logger's level to DEBUG
        self.setLevel(logging.DEBUG)

        self.temp_logger = None  # We no longer need the temporary logger

    def set_level(self, level):
        """Set the level for this logger and all its handlers.

        Args:
            level (str or int): The level to set. Allowed levels are: DEBUG, INFO,
                                WARNING, ERROR, and CRITICAL.
        """
        try:
            # If level is a string, convert it to the corresponding integer value
            if isinstance(level, str):
                level = logging._nameToLevel.get(level.upper(),
                                                LogLevel[level.upper()].value)
            # Set the logger's level
            self.setLevel(level)
            # Also set the level for all handlers
            for handler in self.handlers:
                handler.setLevel(level)
        except Exception as e:
            # If an error occurs, log it
            self.error(f"Error while changing the logger's level: {e}")

    def close(self):
        """Close all handlers associated with this logger.
        """
        for handler in self.handlers:
            # Flush the handler to make sure all logging output has been emitted
            handler.flush()
            # Close the handler
            handler.close()
            # Remove the handler from this logger
            self.removeHandler(handler)
        # Clear the handlers list
        self.handlers = []


if __name__ == '__main__':
    # Ejemplo de uso
    logger = CustomLogger('MyLogger')
    logger.debug('Mensaje de debug')
    logger.info('Mensaje informativo')
    logger.warning('Mensaje de advertencia')
    logger.error('Mensaje de error')
    logger.critical('Mensaje critico')

    # Cambiar el nivel de log
    logger.set_level('INFO')
    logger.debug('Este mensaje de debug no se mostrara.')
    logger.info('Este mensaje informativo sí se mostrara.')
    logger.close()