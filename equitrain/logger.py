import logging
import os
from contextlib import contextmanager


class FileLogger:
    LOG_LEVELS = {
        2: logging.INFO,
        1: logging.WARNING,
        0: logging.ERROR,
    }

    def __init__(
        self,
        enable_logging=False,
        log_to_file=False,
        output_dir=None,
        logger_name='EqLog',
        verbosity=0,
    ):
        """
        Initialize the FileLogger.

        Parameters:
            enable_logging (bool): Flag to enable or disable logging for this instance.
            log_to_file (bool): Indicates if this instance should write logs to file.
            output_dir (str): Directory to store log files.
            logger_name (str): Name for the logger.
            verbosity (int): Verbosity level (0 = minimal, 1 = normal, 2 = warning).
        """
        self.enable_logging = enable_logging
        self.output_dir = output_dir
        self.verbosity = verbosity

        if enable_logging:
            self.logger_name = logger_name
            self.logger = self._setup_logger(log_to_file)
        else:
            self.logger_name = None
            self.logger = NoOp()

    def _setup_logger(self, log_to_file):
        logger = logging.getLogger(self.logger_name)
        log_level = self.LOG_LEVELS.get(self.verbosity, logging.INFO)
        logger.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s - %(message)s')

        if self.output_dir and log_to_file:
            os.makedirs(self.output_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(self.output_dir, 'trainer.log')
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.propagate = False
        return logger

    def log(self, level, message):
        """
        Log a message based on the verbosity level.

        Parameters:
            level (int): Verbosity level (0 = INFO, 1 = WARNING, 2 = DEBUG).
            message (str): Message to log.
        """
        if self.enable_logging and level in self.LOG_LEVELS:
            log_method = {
                logging.INFO: self.logger.info,
                logging.WARNING: self.logger.warning,
                logging.ERROR: self.logger.error,
            }.get(self.LOG_LEVELS[level], self.logger.info)
            log_method(message)

    @contextmanager
    def use(self):
        """Context manager for FileLogger."""
        try:
            yield self
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up handlers to avoid duplication or memory leaks."""
        if self.enable_logging:
            for handler in self.logger.handlers:
                handler.close()
            self.logger.handlers.clear()


class NoOp:
    def __getattr__(self, name):
        def no_op(*args, **kwargs):
            pass

        return no_op
