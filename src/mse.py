import os
import logging
import matplotlib
import pandas as pd
from utils.path_utils import PathUtils
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class MeanSquaredError:
    def __init__(self, _final_df) -> None:
        self._final_df = _final_df
        # calculate the Mean Squared Error between the true and predicted magnitude for the frequency w.r.t
        # as a function of lambda
        self.calculate_mse()
        
    def calculate_mse(self):
        _mse_list = []
        _mse_form_list = []  # MSE calculated via formula
        
        # extract the 'Lambda'values from dataframe
        _res_df = self._final_df.copy(deep=True)
        lmda_lst = _res_df.apply(lambda row: row['Lambda'], axis=1).tolist()
        
        # remove the duplicate lambda values from the list
        lmda_lst = list(dict.fromkeys(lmda_lst))
        
        for p in lmda_lst:
            rslt_frf_df = _res_df[_res_df['Lambda'].isin([p])]
            true_mag = rslt_frf_df['Org_Mag'].to_list()
            pred_mag = rslt_frf_df['Pred_Mag'].to_list()
            _mse = mean_squared_error(true_mag, pred_mag)
            _mse_formula = self._calc_mse_via_formula(true_mag, pred_mag)
            # logging.info(f'Lambda : {p} , MSE = {_mse}')
            _mse_list.append(_mse)
            _mse_form_list.append(_mse_formula)
        # endfor
        
        max_value = max(_mse_list)
        max_index = _mse_list.index(max_value)
        logging.info(
            "MSE calculation completed, check the csv file for detailed results")
        logging.info(
            f'Max Value of MSE : {max_value}, Lambda : {lmda_lst[max_index]}')
        fig, ax = plt.subplots()
        ax.plot(lmda_lst, _mse_list)

        # set labels for x and y axis
        ax.set_xlabel('Lambda')
        ax.set_ylabel('Mean Squared Error (MSE)')

        # set title for the plot
        ax.set_title('MSE Vs Lambda Parameter')
        
        results_dir = PathUtils().get_results_directory()
        mse_plt = os.path.join(results_dir, 'MSE_Plot.png')
        p_name = os.path.realpath(mse_plt)
        fig.savefig(p_name, bbox_inches='tight', dpi=150)
        fig.clf()
        fig.clear()
        matplotlib.pyplot.close()
        
        _mse_df = pd.DataFrame(
            list(zip(lmda_lst, _mse_list, _mse_form_list)), columns=['Lambda', 'MSE', 'MSE_formula'])
        mse_file_name = os.path.join(results_dir, 'mse_df.csv')
        _mse_df.to_csv(os.path.realpath(mse_file_name))
        
    def _calc_mse_via_formula(self, t_mag, p_mag) -> float:
        n = len(t_mag)
        _sum = 0
        for i in range(0, n):
            diff = t_mag[i] - p_mag[i]  # observed - predicted
            sq_diff = diff**2  # square of the diff
            _sum = _sum + sq_diff  # sum these difference for the entire 'n'
            logging.debug(f"TrueMag : {t_mag[i]}, PredMag : {p_mag[i]}, sq_diff:{sq_diff}, Sum: {_sum}")
        # endfor
        _MSE = _sum/n
        return _MSE