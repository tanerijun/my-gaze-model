import logging

def get_logger(log_file=None):
    """
    Returns a logger that logs to both console and, optionally, a file.
    Args:
        log_file (str, optional): Path to the log file.
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger('new_gaze_logger')

    # Prevent adding handlers multiple times if the function is called again
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
