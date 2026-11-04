
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Configures and returns a centralized logger."""

    logger = logging.getLogger(name)

    # Only configure if it doesn't already have handlers.
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Standardized format with source code location.
        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | "
                "%(levelname)-8s | "
                "%(name)s | "
                "%(filename)s:%(lineno)d | "
                "%(funcName)s() | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console output.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)

        logger.addHandler(console_handler)

        # Prevent duplicate propagation to the root logger.
        logger.propagate = False

    return logger
