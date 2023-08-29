import os
import json

class PathUtils:
    def __init__(self) -> None:
        pass

    def get_current_directory(self) -> str:
        self.directory_path = os.getcwd()
        
    def get_data_directory(self) -> str:
        data_directory = os.path.join(self.get_current_directory(), 'data')
        return data_directory

    def get_rawdata_directory(self) -> str:
        raw_data_directory = os.path.join(self.get_data_directory, 'raw')
        return raw_data_directory
    
    def get_results_directory(self) -> str:
        results_directory = os.path.join(self.get_current_directory(), 'results')
        try:
            os.makedirs(results_directory)
        except FileExistsError:
            # results folder/directory already exists
            pass
        return results_directory
    
    def get_log_directory(self) -> str:
        log_directory = os.path.join(self.get_current_directory(), 'logs')
        try:
            os.makedirs(log_directory)
        except FileExistsError:
            # logs folder/directory already exists
            pass
        return log_directory
    
    def get_plots_directory(self) -> str:
        plot_directory = os.path.join(self.get_current_directory(), 'plots')
        try:
            os.makedirs(plot_directory)
        except FileExistsError:
            # plots directory already exists
            pass
        return plot_directory
    
    def get_circle_plots_directory(self) -> str:
        circle_plot_directory = os.path.join(self.get_plots_directory(), 'circle_plots')
        try:
            os.makedirs(circle_plot_directory)
        except FileExistsError:
            # plots directory already exists
            pass
        return circle_plot_directory
    
    def get_models_dir_path(self) -> str:
        model_directory = os.path.join(self.get_current_directory(), 'models')
        try:
            os.makedirs(model_directory)
        except FileExistsError:
            # model directory already exists
            pass
        return model_directory
    
    def create_freq_dir(self, frq_name)->str:
        _fq_dir = f'frq_{frq_name}'
        _model_dir = os.path.join(self.get_circle_plots_directory, _fq_dir)
        try:
            os.makedirs(_model_dir)
        except FileExistsError:
            # model directory already exists
            pass
        return _model_dir
    
    def _read_config(self):
        with open('config.json', 'r') as f:
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
        with open('config.json', 'r+') as _file:
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