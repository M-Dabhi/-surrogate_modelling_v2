import argparse
import logging

from utils.path_utils import PathUtils
from utils.configs import Configs
from helper.base_logger import BaseLogger

from data_parser import DataParser
from data_visualizer import DataVisualizer
from least_squares_circle import LeastSquaresCircle
from least_squares_ellipse import LeastSquaresEllipse
from quadratic_spline import QuadraticSpline
from cubic_spline import CubicSpline
from quartic_spline import QuarticSpline

from data_modeling import DataModeling
from magnitude_plot import MagnitudePlot

log = logging.getLogger(__name__)

def main(log_level: str, draw_plot: str):
    BaseLogger(log_level)

    k = 'draw_plots'
    Configs()._write_to_config(k, draw_plot)

    parse_freq_data()  # parse the freqency data.. create dataframe..
    visualize_data()
    #create_model_predict()
    #least_squares_ellipse()
    #quadratic_spline()
    cubic_spline()
    #quartic_spline()
    
def parse_freq_data():
    dataparser = DataParser()
    if Configs()._parse_data() == True:
        log.info("Processing the frequency data ...")
        dataparser.parse_freq_data()
        dataparser.process_data()
        dataparser.create_data_frame()
        # data is parsed and processed, one can reuse the created dataframe in the next
        # subsequent runs , care should be taken to make the changes in config.json file to
        # process the data to "TRUE", if the data has been changed / deleted.
        Configs()._write_to_config('parse_data', 'False')
        Configs()._write_to_config('process_data', 'False')
    else:
        log.info("data is available...fast forwarding...")
        
def visualize_data():
    # draw the plot only the config value to draw is TRUE
    #DataVisualizer().get_polar_plot_fig_wrt_freq()
    if Configs()._draw_plots() == True:
        _dv = DataVisualizer()
        _dv.plot_frf_data()
        _dv.draw_mag_parameter_plot()
        _dv.draw_polar_plot()
        
def create_model_predict():
    _lsc = LeastSquaresCircle()
    _lsc.process_circle_extraction()
    _df_list = _lsc._get_df_list()

    #log.info(f"Df list in main : {_df_list}")
    _dm = DataModeling()
    _dm.set_df_list(_df_list)
    _dm.model_data()
    _final_df = _dm.get_final_df()
    MagnitudePlot(_final_df)
    #CircleParamPlot(_df_list)


    # if Configs()._draw_plots() == 'True':
    #     _final_df = _dm.get_final_df()
    #     MagnitudePlot(_final_df)
    
def least_squares_ellipse():
    _lse = LeastSquaresEllipse()
    _lse.process_ellipse_extraction()
    
def quadratic_spline():
    _qs = QuadraticSpline()
    _qs.process_spline_extraction()
    
def cubic_spline():
    _cs = CubicSpline()
    _cs.process_spline_extraction()
    
def quartic_spline():
    _cs = QuarticSpline()
    _cs.process_spline_extraction()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Surrogate Modeling Using Complex Valued Frequency Domain Simulation Data')
    parser.add_argument('-l', '--log_level', type=str, default='info',
                        choices=['debug', 'info'], help='Change the log level to Debug or Info')
    parser.add_argument('-dp', '--draw_plot', type=str, default= 'False',
                        choices=['True', 'False'],
                        help='True if the plots (models, relationship plots) as to be\
                              saved into the local directory, False otherwise')
    args = parser.parse_args()
    main(log_level=args.log_level.upper(), draw_plot=args.draw_plot)