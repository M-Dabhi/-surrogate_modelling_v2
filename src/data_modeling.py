import itertools
import os
from utils.path_utils import PathUtils
from utils.configs import Configs
import math
import logging
import pandas as pd
import numpy as np

import time
from humanfriendly import format_timespan
import matplotlib.pyplot as plt
from matplotlib import rcParams
from plot_prediction import PlotPrediction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# import random
# random.seed(100)

RADIUS = 'Radius'
ANGLE = 'Angle'
X_CENTER = 'x_center'
Y_CENTER = 'y_center'
LIST_SIZE = 1
REF_MODEL_SCORE = 0.95

log = logging.getLogger(__name__)

class DataModeling:
    
    def __init__(self) -> None:
        self.res_freqs = []
        self._model_dir = PathUtils().get_models_dir_path()
        self._plot = Configs()._draw_plots()
        #self._df_list = df

        self.freqncy_list = []
        self.lambda_list = []
        # original data attributes
        self.orig_x_list = []
        self.orig_y_list = []
        self.orig_mag_list = []
        # predicted data attributes
        self.pred_x_list = []
        self.pred_y_list = []
        self.pred_mag_list = []
        # newly created dataframe with original and predicted values
        self.new_df_list = []
        
        # based on the experiments
        self.radius_degree = 9
        self.phase_degree = 4
        self.xcenter_degree = 10
        self.ycenter_degree = 9
        
    def set_df_list(self, df):
        self._df_list = df

    def model_data(self) -> None:
        log.info('Modelling the data...')
        #self.extract_dataframe_info()
        wrking_lmda_list = self.extract_fewer_samples()
        self.extract_df_info_fewer_samples(wrking_lmda_list)

    def reset_polynomial_degree(self):
        # based on the experiments
        self.radius_degree = 10
        self.phase_degree = 4
        self.xcenter_degree = 12
        self.ycenter_degree = 10

    def extract_fewer_samples(self):
        """
        1. Pick fewer lambda samples uniformly from the dataframe for pick eg. 7 values
        in between 7-8 and so on until 11. At the moment the data contains lambda samples
        are 50 per range (between 7-8). Train the model with these fewer samples
        and interpolate the data for other lambda values.
        """
        # _wrking_df = self._df_list[:]
        log.info(f"Length of the df list : {len(self._df_list)}")
        if(len(self._df_list) > 1):
            _wrking_df = pd.concat(self._df_list)
        else:
            _wrking_df = self._df_list[0]
        #log.info(f'working df : {_wrking_df}')
        _wrking_df = _wrking_df.sort_values(
            ['Lambda', 'Frequency'], ascending=[True, True])
        _wrking_df.reset_index(drop=True, inplace=True)
        lmda_lst = _wrking_df.apply(lambda row: row['Lambda'], axis=1).tolist()

        # remove the duplicate lambda values from the list
        lmda_lst = list(dict.fromkeys(lmda_lst))
        lmda_lst_cp = lmda_lst[:]
        lmbda_sublists = []

        for _ in range(0, 4): # here we are looping four times because we have ranges, 7-8, 8-9, 9-10, 10-11
            lmda_lst_cp.sort()
            sub_list = lmda_lst_cp[:50]
            sub_list.sort()
            lmda_lst_cp = list(set(lmda_lst_cp)-set(sub_list))
            # 17 elements would be picked
            sub_list = sub_list[0:len(sub_list)-1:3]
            lmbda_sublists.append(sub_list)
        # to avoid extrapolation for the lambda at 11.0 , add the last value to list
        # so that we train the model for this value too.
        # add the last lambda value to list
        lmbda_sublists.append([lmda_lst[-1]])
        # merge all sublists into one , flatten the list of lists into one
        log.debug(f'Lambda Sublists : {lmbda_sublists}')
        merged_list = list(itertools.chain(*lmbda_sublists))
        log.info(f'Length of Lambda List for train/test : {len(merged_list)}')
        return merged_list
    
    # ---------------------------------------------------------------------------------------
    def extract_dataframe_info(self):
        start_time = time.monotonic()
        for _df in self._df_list:
            fq_list = _df['Frequency'].to_list()
            _fq = list(dict.fromkeys(fq_list))

            self.fq = _fq[-1]
            res_df = _df[_df['Frequency'].isin([self.fq])]
            lambda_vals = res_df['Lambda']  # length here is 200

            # radius_deg = res_df['radius_degree']
            # radius_deg = list(dict.fromkeys(radius_deg))
            # self.radius_degree = radius_deg[-1] #one value

            # ph_deg = res_df['phase_degree']
            # ph_deg = list(dict.fromkeys(ph_deg))
            # self.phase_degree = ph_deg[-1] #one value

            # xc_deg = res_df['xc_degree']
            # xc_deg = list(dict.fromkeys(xc_deg))
            # self.xcenter_degree = xc_deg[-1] #one value

            # yc_deg = res_df['yc_degree']
            # yc_deg = list(dict.fromkeys(yc_deg))
            # self.ycenter_degree = yc_deg[-1] #one value

            #log.info(f'deg r = {self.radius_degree}, ph = {self.phase_degree}, xc = {self.xcenter_degree}, yc = {self.ycenter_degree}')

            self.model_radius(lambda_vals, res_df[RADIUS])
            self.model_phase(lambda_vals, res_df[ANGLE])
            self.model_xcenter(lambda_vals, res_df[X_CENTER])
            self.model_ycenter(lambda_vals, res_df[Y_CENTER])

            # models are prepared.. it's time for some predictions ;)
            log.debug(f'Predictions for the frequency {self.fq} is,')
            for l_val in lambda_vals:
                self.predict_radius(l_val)
                self.predict_phase(l_val)
                self.predict_Xc(l_val)
                self.predict_Yc(l_val)
                self.extract_predicted_xy_coord(l_val, res_df)

            # create dataframe with the desired for regression modeling
            _new_df = pd.DataFrame(list(zip(
                self.freqncy_list, self.lambda_list,
                self.orig_x_list, self.orig_y_list, self.orig_mag_list,
                self.pred_x_list, self.pred_y_list, self.pred_mag_list)),
                columns=['Frequency', 'Lambda', 'Org_X', 'Org_Y', 'Org_Mag', 'Pred_X', 'Pred_Y', 'Pred_Mag'])
            self.clear_lists()
            self.new_df_list.append(_new_df)
        # endfor
        end_time = time.monotonic()
        log.info(
            f'Predictions finished in {format_timespan(end_time - start_time)}')
        # concat the dataframes in the list and create a single dataframe with the final data for further plotting
        self.final_df = pd.concat(self.new_df_list)
        self.final_df = self.final_df.sort_values(
            ['Lambda', 'Frequency'], ascending=[True, True])
        self.final_df.reset_index(drop=True, inplace=True)
        df_name = os.path.join(PathUtils.get_results_directory, 'final_df.csv')
        self.final_df.to_csv(os.path.realpath(df_name))
        # Plot the True and Predicted Magnitude values w.r.t frequency as function of lambda
        PlotPrediction(self.final_df)
        
    # -------------------------------------------------------------------------------------------

    def extract_df_info_fewer_samples(self, _lambda_list):
        start_time = time.monotonic()
        for _df in self._df_list:
            fq_list = _df['Frequency'].to_list()
            _fq = list(dict.fromkeys(fq_list))

            self.fq = _fq[-1]
            res_df = _df[_df['Frequency'].isin([self.fq])]
            # use the res_df to extract the fewer lambda samples
            _wrking_df = res_df[res_df['Lambda'].isin(_lambda_list)]

            lambda_vals = _wrking_df['Lambda']  # length here is 69
            org_lambda_list = res_df['Lambda']  # length here is 200

            radius_deg = _wrking_df['radius_degree']
            radius_deg = list(dict.fromkeys(radius_deg))
            self.radius_degree = radius_deg[-1] #one value

            ph_deg = _wrking_df['phase_degree']
            ph_deg = list(dict.fromkeys(ph_deg))
            self.phase_degree = ph_deg[-1] #one value

            xc_deg = _wrking_df['xc_degree']
            xc_deg = list(dict.fromkeys(xc_deg))
            self.xcenter_degree = xc_deg[-1] #one value

            yc_deg = _wrking_df['yc_degree']
            yc_deg = list(dict.fromkeys(yc_deg))
            self.ycenter_degree = yc_deg[-1] #one value

            #log.info(f'deg r = {self.radius_degree}, ph = {self.phase_degree}, xc = {self.xcenter_degree}, yc = {self.ycenter_degree}')

            # training is done on the fewer lambda samples
            self.model_radius(lambda_vals, _wrking_df[RADIUS])
            self.model_phase(lambda_vals, _wrking_df[ANGLE])
            self.model_xcenter(lambda_vals, _wrking_df[X_CENTER])
            self.model_ycenter(lambda_vals, _wrking_df[Y_CENTER])

            # models are prepared.. it's time for some predictions ;)
            log.debug(f'Predictions for the frequency {self.fq} is,')
            for l_val in org_lambda_list:  # predict for all the lambda values
                self.predict_radius(l_val)
                self.predict_phase(l_val)
                self.predict_Xc(l_val)
                self.predict_Yc(l_val)
                self.extract_predicted_xy_coord(l_val, res_df)
            # create dataframe with the desired for regression modeling

            _new_df = pd.DataFrame(list(zip(
                self.freqncy_list, self.lambda_list,
                self.orig_x_list, self.orig_y_list, self.orig_mag_list,
                self.pred_x_list, self.pred_y_list, self.pred_mag_list)),
                columns=['Frequency', 'Lambda', 'Org_X', 'Org_Y', 'Org_Mag', 'Pred_X', 'Pred_Y', 'Pred_Mag'])
            self.clear_lists()
            self.new_df_list.append(_new_df)
        # endfor
        end_time = time.monotonic()
        log.info(
            f'Predictions finished in {format_timespan(end_time - start_time)}')
        # concat the dataframes in the list and create a single dataframe with the final data for further plotting
        self.final_df = pd.concat(self.new_df_list)
        self.final_df = self.final_df.sort_values(
            ['Lambda', 'Frequency'], ascending=[True, True])
        self.final_df.reset_index(drop=True, inplace=True)
        df_name = os.path.join(PathUtils.get_results_directory, 'final_df.csv')
        self.final_df.to_csv(os.path.realpath(df_name))
        # Plot the True value and Predicted Magnitude values w.r.t frequency as function of lambda
        PlotPrediction(self.final_df)
        
    # ----------------------------------------------------------------------------------------------

    def clear_lists(self) -> None:
        self.orig_x_list.clear()
        self.orig_y_list.clear()
        self.orig_mag_list.clear()
        self.pred_x_list.clear()
        self.pred_y_list.clear()
        self.pred_mag_list.clear()
        self.freqncy_list.clear()
        self.lambda_list.clear()

# --------------------------------------------------------------------------------------
    """
    Modeling Radius using polynomial regression fit.
    """

    def model_radius(self, lmbda, radii) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(radii)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=10)
        if self._plot:
            rcParams['axes.spines.top'] = False
            rcParams['axes.spines.right'] = False

            _, ax1 = plt.subplots(1, 1)

            ax1.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
            ax1.scatter(x_train, y_train, c='#8acfd4', label='Training data')
            ax1.legend(loc="upper left")

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        self.radii_poly = PolynomialFeatures(degree=self.radius_degree)
        x_poly = self.radii_poly.fit_transform(x_train)

        self.reg_radius = LinearRegression()
        res = self.reg_radius.fit(x_poly, y_train)

        xtest_poly = self.radii_poly.fit_transform(x_test)
        y_pred = self.reg_radius.predict(xtest_poly)
        model_score = res.score(x_poly, y_train)
        log.debug(f'Model Score (Radius) : {model_score}')
        if self._plot:
            ax1.set_title('Regression w.r.t Radius')
            ax1.set_xlabel('Lambda')
            ax1.set_ylabel('Radius')
            ax1.plot(x_train, self.reg_radius.predict(x_poly),
                     c='blue', label='Predicted Line')
            ax1.legend(loc="best")
            f_name = os.path.join(self._model_dir, f'Radius[Model]_{self.fq}.png') # '{0}/' + f'Radius[Model]_{self.fq}.png'
            plt.savefig(os.path.realpath(f_name))
            plt.close()
        return model_score
    
    # ---------------------------------------------------------------------------

    def predict_radius(self, lambda_val):
        pred_r = self.reg_radius.predict(
            self.radii_poly.transform([[lambda_val]]))

        pred_r = list(pred_r.flatten())
        self.R = pred_r[-1]
        log.debug(f'Predicted Radius : {self.R}')
        
    # --------------------------------------------------------------------------------------
    """
    Modeling Phase using polynomial regression fit.
    """

    def model_phase(self, lmbda, angle) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(angle)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=10)

        if self._plot:
            rcParams['axes.spines.top'] = False
            rcParams['axes.spines.right'] = False

            _, ax2 = plt.subplots(1, 1)

            ax2.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
            ax2.scatter(x_train, y_train, c='#8acfd4', label='Training data')
            ax2.legend(loc="upper left")

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        self.ph_poly = PolynomialFeatures(degree=self.phase_degree)
        x_poly = self.ph_poly.fit_transform(x_train)

        self.reg_phase = LinearRegression()
        res = self.reg_phase.fit(x_poly, y_train)
        
        model_score = res.score(x_poly, y_train)
        log.debug(f'Model Score (Angle) : {model_score}')
        # pred_ph_deg = math.degrees(Ph)
        # v2  = np.rad2deg(Ph)
        # print('predicted angle in degrees : {}, {}'.format(pred_ph_deg, v2))
        if self._plot:
            ax2.set_title('Regression w.r.t Angle')
            ax2.set_xlabel('Lambda')
            ax2.set_ylabel('Angle')
            ax2.plot(x_train, self.reg_phase.predict(x_poly),
                     c='blue', label='Predicted Line')
            ax2.legend(loc="upper right")

            f_name = os.path.join(self._model_dir, f'Phase[Model]_{self.fq}.png')  #'{0}/' + f'Phase[Model]_{self.fq}.png'
            plt.savefig(os.path.realpath(f_name))
            plt.close()
        return model_score
    
    # ---------------------------------------------------------------------------
    def predict_phase(self, lambda_val):
        pred_an = self.reg_phase.predict(
            self.ph_poly.transform([[lambda_val]]))

        pred_an = list(pred_an.flatten())
        self.Ph = pred_an[-1]
        log.debug(f'Predicted Angle : {self.Ph}')
        
    # --------------------------------------------------------------------------------------
    """
    Modeling X- Center using polynomial regression fit.
    """

    def model_xcenter(self, lmbda, x_center) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(x_center)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=10)

        if self._plot:
            rcParams['axes.spines.top'] = False
            rcParams['axes.spines.right'] = False

            _, ax3 = plt.subplots(1, 1)

            ax3.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
            ax3.scatter(x_train, y_train, c='#8acfd4', label='Training data')
            ax3.legend(loc="upper left")

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        self.poly_xc = PolynomialFeatures(degree=self.xcenter_degree)
        x_poly = self.poly_xc.fit_transform(x_train)

        self.reg_xc = LinearRegression()
        res = self.reg_xc.fit(x_poly, y_train)
        model_score = res.score(x_poly, y_train)
        log.debug(f'Model Score (X-Center) : {model_score}')
        if self._plot:
            ax3.set_title('Regression w.r.t X_Center')
            ax3.set_xlabel('Lambda')
            ax3.set_ylabel('X_Center')
            ax3.plot(x_train, self.reg_xc.predict(x_poly),
                     c='blue', label='Predicted Line')
            ax3.legend(loc="upper right")

            f_name = os.path.join(self._model_dir, f'X-Coordinate[Model]_{self.fq}.png')  # '{0}/' + f'X-Coordinate[Model]_{self.fq}.png'
            plt.savefig(os.path.realpath(f_name))
            plt.close()
        return model_score

# ---------------------------------------------------------------------------

    def predict_Xc(self, lambda_val):
        pred_Xc = self.reg_xc.predict(self.poly_xc.transform([[lambda_val]]))
        pred_Xc = list(pred_Xc.flatten())
        self.Xc = pred_Xc[-1]
        log.debug(f'Predicted X-Center : {self.Xc}')

# --------------------------------------------------------------------------------------
    """
    Modeling Y-Center using polynomial regression fit.
    """

    def model_ycenter(self, lmbda, y_center) -> None:
        X = np.array(lmbda).reshape(-1, 1)
        y = np.array(y_center)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=10)
        if self._plot:
            rcParams['axes.spines.top'] = False
            rcParams['axes.spines.right'] = False

            _, ax4 = plt.subplots(1, 1)

            ax4.scatter(x_test, y_test, c='#edbf6f', label='Testing data')
            ax4.scatter(x_train, y_train, c='#8acfd4', label='Training data')
            ax4.legend(loc="upper left")

        x_train = x_train.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

        y_train = y_train[x_train[:, 0].argsort()]
        x_train = x_train[x_train[:, 0].argsort()]

        self.poly_yc = PolynomialFeatures(degree=self.ycenter_degree)
        x_poly = self.poly_yc.fit_transform(x_train)

        self.reg_yc = LinearRegression()
        res = self.reg_yc.fit(x_poly, y_train)

        model_score = res.score(x_poly, y_train)
        log.debug(f'Model Score (Y-Center) :  {model_score}')
        if self._plot:
            ax4.set_title('Regression w.r.t y_center')
            ax4.set_xlabel('Lambda')
            ax4.set_ylabel('Y_Center')
            ax4.plot(x_train, self.reg_yc.predict(x_poly),
                     c='blue', label='Predicted Line')
            ax4.legend(loc="upper left")

            f_name = os.path.join(self._model_dir, f'Y-coordinate[Model]_{self.fq}.png')  # '{0}/' + f'Y-coordinate[Model]_{self.fq}.png'
            plt.savefig(os.path.realpath(f_name))
            plt.close()
        return model_score

# ---------------------------------------------------------------------------
    def predict_Yc(self, lambda_val):
        pred_Yc = self.reg_yc.predict(self.poly_yc.transform([[lambda_val]]))
        pred_Yc = list(pred_Yc.flatten())
        self.Yc = pred_Yc[-1]
        log.debug(f'Predicted Y-Center : {self.Yc}')

# ---------------------------------------------------------------------------
    def extract_predicted_xy_coord(self, l_val, _df):
        a = math.cos(self.Ph)
        b = math.sin(self.Ph)
        X_coord = self.Xc + (self.R * a)
        Y_coord = self.Yc + (self.R * b)
        # for now the list size is 1
        # _df = self._df_list[-1]
        _dframe = _df[_df.Lambda == l_val]
        org_crd = _dframe['coordinates'].to_list()
        log.debug('Freq: {0} , Lambda: {1}'.format(
            _dframe.iloc[0]['Frequency'], _dframe.iloc[0]['Lambda']))  # .iloc[0]['A']
        log.debug(f'\tTrue Coordinate : {org_crd[-1]}')
        self.freqncy_list.append(self.fq)
        self.lambda_list.append(l_val)
        if len(org_crd) == LIST_SIZE:
            coord_tup = org_crd[-1]
            org_x = coord_tup[0]
            org_y = coord_tup[1]
            self.orig_x_list.append(org_x)
            self.orig_y_list.append(org_y)
            org_cn = complex(org_x, org_y)
            org_mag = abs(org_cn)
            self.orig_mag_list.append(org_mag)
        else:
            log.error("Please check the coordinates extraction.")

        log.debug(f'Predicted Coordinate : ({X_coord}, {Y_coord})')
        logging.debug('-------------------------------------------------------')
        self.pred_x_list.append(X_coord)
        self.pred_y_list.append(Y_coord)
        cn = complex(X_coord, Y_coord)
        mag = abs(cn)
        self.pred_mag_list.append(mag)

# ----------------------------------------------------------------------------------
    def get_final_df(self):
        return self.final_df