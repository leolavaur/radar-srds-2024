import logging
from functools import wraps

from flwr.common.logger import logger as flwr_logger

logger = logging.getLogger("trustfids")
logger.setLevel(logging.INFO)

flwr_logger.removeHandler(flwr_logger.handlers[0])


def logged(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        logger.debug(
            "Called `%s` with args: %s",
            function.__name__,
            {**dict(zip(function.__code__.co_varnames, args)), **kwargs},
        )

        result = function(*args, **kwargs)
        return result

    return wrapper


# flwr_logger.addHandler(logger.handlers[0])


# class CustomFormatter(logging.Formatter):

#     grey = "\x1b[38;20m"
#     yellow = "\x1b[33;20m"
#     red = "\x1b[31;20m"
#     bold_red = "\x1b[31;1m"
#     reset = "\x1b[0m"
#     format = (
#         "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
#     )

#     FORMATS = {
#         logging.DEBUG: grey + format + reset,
#         logging.INFO: grey + format + reset,
#         logging.WARNING: yellow + format + reset,
#         logging.ERROR: red + format + reset,
#         logging.CRITICAL: bold_red + format + reset,
#     }

#     def format(self, record):
#         log_fmt = self.FORMATS.get(record.levelno)
#         formatter = logging.Formatter(log_fmt)
#         return formatter.format(record)


# def set_level(level: int) -> None:
#     """Set the logging level for all loggers.

#     Args:
#         level (int): Logging level.
#     """
#     loggers = [
#         logging.getLogger(name)
#         for name in logging.root.manager.loggerDict
#         if name == "__main__" or name.startswith("trustfids")
#     ]

#     for l in loggers:
#         l.setLevel(logging.DEBUG)
