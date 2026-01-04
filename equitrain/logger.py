import logging
import os
from contextlib import contextmanager
from pathlib import Path


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
        stream=True,
        log_suffix=None,
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
        self.logger_name = logger_name
        self._stream = bool(stream)
        self._log_suffix = '' if log_suffix is None else str(log_suffix)
        self.logger = self._setup_logger(log_to_file)

    def _setup_logger(self, log_to_file):
        logger = logging.getLogger(self.logger_name)
        log_level = self.LOG_LEVELS.get(self.verbosity, logging.INFO)
        logger.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s - %(message)s')

        if self.output_dir and log_to_file:
            os.makedirs(self.output_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(self.output_dir, f'trainer{self._log_suffix}.log')
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if self._stream:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        logger.propagate = False
        return logger

    def log(self, level, message, force=False):
        """
        Log a message based on the verbosity level.

        Parameters:
            level (int): Verbosity level (0 = INFO, 1 = WARNING, 2 = DEBUG).
            message (str): Message to log.
        """
        if (self.enable_logging or force) and level in self.LOG_LEVELS:
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
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()


def ensure_output_dir(path: str | None) -> None:
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def init_logger(
    args,
    *,
    backend_name: str,
    enable_logging: bool,
    log_to_file: bool,
    output_dir: str | None,
    stream: bool = True,
    log_suffix: str | None = None,
) -> FileLogger:
    return FileLogger(
        enable_logging=enable_logging,
        log_to_file=log_to_file,
        output_dir=output_dir,
        logger_name=f'Equitrain[{backend_name}]',
        verbosity=getattr(args, 'verbose', 0),
        stream=stream,
        log_suffix=log_suffix,
    )
