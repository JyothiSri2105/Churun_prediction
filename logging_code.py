import logging
import os


def setup_logging(name):

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_folder = "logs"

    os.makedirs(log_folder, exist_ok=True)

    log_file = os.path.join(log_folder, f"{name}.log")

    handler = logging.FileHandler(log_file)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.propagate = False

    return loggerimport logging
import os


def setup_logging(name):

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_folder = "logs"

    os.makedirs(log_folder, exist_ok=True)

    log_file = os.path.join(log_folder, f"{name}.log")

    handler = logging.FileHandler(log_file)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.propagate = False

    return logger