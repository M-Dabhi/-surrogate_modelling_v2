import os
import logging
from utils.path_utils import PathUtils
import time

FORMAT_STR = '%(asctime)s [%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class BaseLogger:
    def __init__(self, log_level) -> None:
        root = logging.getLogger()
        root.setLevel(logging.NOTSET)

        # setup logging to console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # setup log format
        log_formatter = logging.Formatter(FORMAT_STR, datefmt=DATE_FORMAT)
        console_handler.setFormatter(log_formatter)

        # add console handler to root logger
        root.addHandler(console_handler)

        # set log file path
        log_directory = PathUtils.get_log_directory()
        file_name = f'{time.strftime("%Y%m%d_%H%M%S")}_log.txt'
        file_path = os.path.join(log_directory, file_name)
        
        # setup log file handler
        file_handler = logging.FileHandler(filename=os.path.realpath(file_path), mode='w', delay=True)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(log_level)
        
        # add log file handler to root handler
        root.addHandler(file_handler)
        
