import os
import json
from helper.singleton import Singleton

class PathUtils(metaclass=Singleton):
    def __init__(self) -> None:
        pass

    def get_current_directory(self) -> str:
        self.directory_path = os.path.dirname(os.getcwd())  # Here we are getting main directory i.e., surrogate_modelling_v2 as in current working directory
        return self.directory_path
        
    def get_data_directory(self) -> str:
        data_directory = os.path.join(self.get_current_directory(), 'data')
        return data_directory

    def get_rawdata_directory(self) -> str:
        raw_data_directory = os.path.join(self.get_data_directory(), 'raw')
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
    
    def create_freq_dir(self, frq_name) -> str :
        _fq_dir = f'frq_{frq_name}'
        _model_dir = os.path.join(self.get_circle_plots_directory(), _fq_dir)
        try:
            os.makedirs(_model_dir)
        except FileExistsError:
            # model directory already exists
            pass
        return _model_dir
    
    def get_ellipse_plots_directory(self) -> str:
        ellipse_plot_directory = os.path.join(self.get_plots_directory(), 'ellipse_plots')
        try:
            os.makedirs(ellipse_plot_directory)
        except FileExistsError:
            # plots directory already exists
            pass
        return ellipse_plot_directory
    
    def get_spline_plots_directory(self, frq_name) -> str:
        _fq_dir = f'frq_{frq_name}'
        spline_plot_directory = os.path.join(self.get_plots_directory(), 'spline_plots', _fq_dir)
        try:
            os.makedirs(spline_plot_directory)
        except FileExistsError:
            # plots directory already exists
            pass
        return spline_plot_directory
    
    def create_freq_dir_for_plots(self, freq) -> str : 
        freq = f'frq_{freq}'
        freq_plot_dir = os.path.join(self.get_plots_directory(), freq)
        try: 
            os.makedirs(freq_plot_dir)
        except FileExistsError:
            # model directory already exists
            pass
        return freq_plot_dir
        
    def get_config_file(self) -> str :
        return os.path.join(os.getcwd(), 'config.json')