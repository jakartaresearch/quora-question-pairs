import logging
import os


def log(path, file):
    """[Create a log file to record the experiment's logs].

    Arguments:
        path {string} -- path to the directory
        file {string} -- file name

    Returns:
        [obj] -- [logger that record logs]
    """
    if not os.path.exists(path):
        os.mkdir(path)
    # check if the file exist
    log_file = os.path.join(path, file)

    logging_format = "%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s"

    # configure logger
    logging.basicConfig(level=logging.DEBUG, format=logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_file, 'w')

    # set the logging level for log file
    handler.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter(logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger
