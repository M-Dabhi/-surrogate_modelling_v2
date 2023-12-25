import json 
import os
from utils.path_utils import PathUtils

class Configs:
    def __init__(self) -> None:
        self.config_file = PathUtils().get_config_file()
    
    def _read_config(self):
        with open(self.config_file, 'r') as f:
            config = json.load(f)
            return config
    
    """
    ref: https://stackoverflow.com/questions/21035762/python-read-json-file-and-modify
    Parameter
    ---------
    key : str
        Key value in the json config
    val : str
        Corresponding value of the key in json config
    """

    def _write_to_config(self, key: str, val: str):
        with open(self.config_file, 'r+') as _file:
            _cfg = json.load(_file)
            _cfg[key] = val  # add the value to config
            _file.seek(0)  # reset the file position to the beginning
            json.dump(_cfg, _file, indent=4)
            _file.truncate()  # remove the remaining part
        
        
    def _draw_plots(self) -> bool:
        _config = self._read_config()
        if _config['draw_plots'] == "True":
            return True
        else:
            return False

    def _parse_data(self) -> bool:
        _config = self._read_config()
        if _config['parse_data'] == "True":
            return True
        else:
            return False