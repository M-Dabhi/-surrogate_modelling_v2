import os
import math
import random
import logging

import time
from humanfriendly import format_timespan

import matplotlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly

from scipy import optimize
from numpy import *
from data_modeling import REF_MODEL_SCORE, DataModeling
from data_parser import DataParser
from utils.path_utils import PathUtils
from utils.configs import Configs

from matplotlib import pyplot as p
import numpy_indexed as npi

import skg

log = logging.getLogger(__name__)

MAX_NEIGH = 25
MAX_STEP_WIDTH = 25
MAX_REF_ANGLE = 21.

# REF_ANGLE_THRESHOLDS = [14, 16]

"""
Circle approximation using Least Squares Method.
Ref: https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
"""


class LeastSquaresCircle:
    def __init__(self) -> None:
        self._plt_dir = PathUtils().get_plots_directory()
        self._df_list = []
        self._plot = Configs()._draw_plots()
        log.info(f"Draw Plots : {self._plot}")
        plotly.io.orca.config.executable = os.path.realpath(os.path.join(PathUtils().get_current_directory, 'orca_app', 'orca.exe')) # r'C:\ProgramData\miniconda3\orca_app\orca.exe'
        plotly.io.orca.config.save()
        self._neigh = 10
        self._step = 4
        self.frq_dist = []
        self.calc_angle = 0.

        self.ref_angle = 19.

        self.ref_angle_thresholds = [self.ref_angle - 0.5, self.ref_angle + 2]

        self.radii_mscore = 0.
        self.phase_mscore = 0.
        self.xc_mscore = 0.
        self.yc_mscore = 0.

        self.reject_model = False

        self._firstxy = 0.
        self._lastxy = 0.

        self.adjusted = False
        self.is_larger = False
        self.is_smaller = False

        self.residu_list = []

        self.res_dict = {'radius': 0., 'center': (0.,0.), 'xy_tuple': (0.,0.), 'residu': 0.}

    """
    Random color generation for plots
    reference : https://stackoverflow.com/a/55828367
    """
    def generate_random_color():
        hex_colors_dic = {}
        rgb_colors_dic = {}
        hex_colors_only = []
        for name, hex in matplotlib.colors.cnames.items():
            hex_colors_only.append(hex)
            hex_colors_dic[name] = hex
            rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)
        # getting random color from list of hex colors
        return random.choice(hex_colors_only)
    
    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return sqrt((self.x-xc)**2 + (self.y-yc)**2)
    
    """
    Computes given x and y points and approximates the best possible
    circle with the given x and y points
    Parameters
    ----------
    x - List
    list of X-coordinates in the which the circle should pass
    y - List
    list of Y-coordinates in the which the circle should pass
    
    Returns
    -------
    Circle center(Xc, Yc) and Radius (R)
    """

    def least_square_circle_extraction(self, x, y):
        # coordinates of the barycenter
        x_m = mean(x)
        y_m = mean(y)

        center_estimate = x_m, y_m
        center_2, ier = optimize.leastsq(self.f_2, center_estimate)

        center_2 = tuple(center_2)
        xc_2, yc_2 = center_2
        Ri_2 = self.calc_R(*center_2)
        R_2 = Ri_2.mean()
        residu_2 = sum((Ri_2 - R_2)**2)
        # log.info("center : {0}, radius : {1}".format(center_2, R_2))
        return xc_2, yc_2, R_2, residu_2
    
    def least_square_circle_fit(self, data):
        r, c = skg.nsphere_fit(data)
        return c[0], c[1], r
    
    def automate_circle_extraction(self, idx: int, freq: float, lmbda: float) -> dict:
        # 1. adjust the neighbours
        # 2. Extract the circle center and radius

        # fl_tup : tuple containing the first and last x,y coordinates of the extracted fre.
        # neigh_info : containing the information regarding the extracted neighbors
        # xy_tuple : x,y coordinate of picked freq. where circle extraction is being processed
        neigh_info, xy_tuple = self.adjust_neighbours_frqs(idx, freq, lmbda)

        # extract the coordinate tuples from the info
        self.x = []
        self.y = []
        freq_pts = []
        coord_list = []
        for _item in neigh_info:
            # 0 : Freq and 1 : Lambda from 2: Coordinates
            coord = tuple(_item[2:])
            # extract the first element of the tuple into list
            self.x.append(coord[0])
            # extract the second element of the tuple into a list
            self.y.append(coord[1])
            coord_list.append(coord)
            freq_pts.append(_item[0])
        # self.x = r_[self.x]  # convert to numpy array
        # self.y = r_[self.y]
        # print('--------------------------------------------------------------------------------------')
        # print("\nExtracted Coordinates {0}: \n|-> Adj. Frequencies : {1}".format(coord, freq_pts))
        log.debug(
            "\n=> Extracted X-Points : {0} \
            \n=> Extracted Y-Points : {1} \
            \n=> Adj. Frequencies : {2}".format(self.x, self.y, freq_pts))

        # Now pass this extracted 3 x-y points and calculate the center and radius
        # h,k,radius = np.float64(three_point_circle(*coord))
        h, k, radius, residu = np.float64(self.least_square_circle_extraction(self.x, self.y))
        #h, k , radius = self.least_square_circle_fit(coord_list)

        self.residu_list.append([lmbda, residu])
        # self._firstxy = fl_tup[0]
        # self._lastxy = fl_tup[1]
        self.calc_angle = self.get_angle(self._firstxy, (h, k), self._lastxy)
        #self.res_dict = {'radius': radius, 'center': (h, k), 'xy_tuple': xy_tuple, 'residu': residu}

        self.res_dict = {'radius': radius, 'center': (h, k), 'xy_tuple': xy_tuple}

        if self.ref_angle_thresholds[0] <= self.calc_angle <= self.ref_angle_thresholds[1]:
            self.adjusted = True
            self.is_smaller = False
            self.is_larger = False
        elif self.calc_angle < self.ref_angle_thresholds[0]:
            self.is_smaller = True
            self.is_larger = False
            self.adjusted = False
        elif self.calc_angle > self.ref_angle_thresholds[1]:
            self.is_larger = True
            self.is_smaller = False
            self.adjusted = False
        
        log.debug(
                f"Calculated angle :-> {self.calc_angle}, Ref. angle :-> {self.ref_angle}, \
                 \nisSmaller :-> {self.is_smaller}, isLarger :-> {self.is_larger}, adjusted :-> {self.adjusted}")
        if (self.adjusted == True):
            pass
        else:
            if self.is_smaller:
                while (self.is_smaller):
                    if (self._step < MAX_STEP_WIDTH):
                        self._step += 1
                    if (self._step >= 10 and self._neigh < MAX_NEIGH):
                        self._neigh += 1
                    self.automate_circle_extraction(idx, freq, lmbda)  # update res_dict
                    # update is_smaller based on the updated value of calc_angle
                    self.is_smaller = self.calc_angle < self.ref_angle_thresholds[0]
                    if self.ref_angle_thresholds[0] <= self.calc_angle <= self.ref_angle_thresholds[1]:
                        self.adjusted = True
                # end while angle has been adjusted
            elif self.is_larger:
                while (self.is_larger):
                    if (self._step < MAX_STEP_WIDTH):
                        self._step -= 1
                    if (self._step >= 10 and self._neigh < MAX_NEIGH):
                        self._neigh += 1
                    self.automate_circle_extraction(idx, freq, lmbda)  # update res_dict
                    # update is_larger based on the updated value of calc_angle
                    self.is_larger = self.calc_angle > self.ref_angle_thresholds[1]
                    if self.ref_angle_thresholds[0] <= self.calc_angle <= self.ref_angle_thresholds[1]:
                        self.adjusted = True
                # end while angle has been adjusted

        # end while angle has been adjusted to approx. near to the ref. angle
        # reset the stepwidth and num of neighbors for the next freq. processing
        #log.debug(f"Exited while, Angle is adapted : stepwidth : {self._step}, Neighbours : {self._neigh}")
        # self.is_smaller = False
        # self.is_larger = False
        
    # -----------------------------------------------------------------------------------------
    def adjust_neighbours_frqs(self, indx, frq, lmbda):
        # self.extract_neighbours()
        # extract the neighbours
        info, xy_tuple = self.extract_neighbours(indx)
        frequency_value = frq
        lambda_value = lmbda
        frq_name = f'Frequency = {frequency_value}'
        lambda_name = f'Lambda = {lambda_value}'
        #log.debug('\n-----------------------------------------------------------------------------------------')
        log.debug('\n=> Extracted Info : {0} \n=> (x,y) : {1}'.format(
            info, xy_tuple))
        log.debug(f'\n=> {frq_name} , {lambda_name}')

        # calculate the distance between the first and last x,y points
        self._firstxy, self._lastxy = tuple(info[0][-2:]), tuple(info[-1][-2:])
        log.debug(f"firstxy : {self._firstxy}, lastxy : {self._lastxy}")

        # x1, y1 = _firstxy[0], _firstxy[1]
        # x2, y2 = _lastxy[0], _lastxy[1]
        # _dist = math.hypot(x2 - x1, y2 - y1)

        fst_lst_tuple = (self._firstxy, self._lastxy)  # info
        log.debug(
            f'Info of the first and last xy coord of current freq : {fst_lst_tuple}')
        return info, xy_tuple
    
    # ------------------------------------------------------------------------------------------
    def pick_every_x_element(self, lst, index):

        pickd_lm = lst[index][1]
        pickd_frq = lst[index][0]

        # print(f'Picked lambda -->  {pickd_lm}, Picked Frq ---> {pickd_frq}')

        # Example data : [34.770000457764, 7.0, 0.12457949668169, -0.48617362976074]
        # Filter based on the lambda value column (index 1)
        filtered_data = list(filter(lambda x: x[1] == pickd_lm, lst))
        lst_cp_r = filtered_data[:]
        lst_cp_l = filtered_data[:]

        # filtered_array = list(filter(lambda x: pickd_frq in x, filtered_data))
        # if filtered_array:
        #     index = filtered_data.index(filtered_array[0])
        # else:
        #     index = -1

        filtered_data_array = np.array(filtered_data)
        searched_values = np.array([lst[index]])
        _result = npi.indices(filtered_data_array, searched_values)

        if _result is not None:
            index = _result[-1]
        else:
            index = -1

        # Check if the given index is valid
        if index < 0 or index >= len(lst_cp_r):
            return None

        log.debug(f'pick_every_x_element: Step Count --> {self._step}')
        # Extract every 10th element from the sublist, starting from the given index
        r_sublist = lst_cp_r[index::self._step]
        # Extract every 10th element to the left sublist, starting from the given index
        sub_slice = lst_cp_l[:index+1]
        l_sublist = sub_slice[::-self._step]

        l_sublist = list(reversed(l_sublist))
        # as picked index is added twice remove it from the left sublist and join with the right sublist
        l_sublist.remove(l_sublist[-1])
        _f_list = l_sublist + r_sublist

        return _f_list
    
    # --------------------------------------------------------------------------------------------------------

    def extract_neighbours(self, idx):
        orgDFs = self.orgDF_list[:]  # copy list into new variable
        _pick = orgDFs[idx]
        extract_xy = _pick[-2:]  # type list
        xy_tuple = tuple(extract_xy)

        # copy list into new variable so we don't change it
        main_list = orgDFs[:]
        _res_list = self.pick_every_x_element(main_list, idx)

        # update the index with the new working list
        _new_idx = _res_list.index(_pick)
        log.debug(f'extract_neighbours: Neigh Count --> {self._neigh}')
        left_slice = _new_idx - self._neigh//2  # start of range

        # account for edges for left slice for the last element in the list
        # internal logic behind the left slice
        # max((0, left_sclice), (len(list) - <req.num.neightbours>)), min(<idx>, <idx-1>),
        # gives the exact neightbours otherwise would get one less neighbour
        left_slice = min(max(0, left_slice), len(_res_list) - self._neigh)

        # extract the right slice range
        right_slice = left_slice + self._neigh  # end of range
        return _res_list[left_slice:right_slice], xy_tuple
    
    # -------------------------------------------------------------------------------------------------------------
    def evaluate_circle_extraction(self, frq, wrk_dict):
        # ----------------------------------------------------------------------------------------------------------------
        """
        - For each 'lambda' value in the working dataframe, we pick the index and 
            extract the adjacent data points for the picked index, which inturn contains 
            the adjacent frequencies data.
        - Pass the extracted real and imaginarly values as tuple to plot the circle 
            which passes through these 3 points.
        - With circle plot we also have circle 'Radius' and 'Center' information.
        """
        # ----------------------------------------------------------------------------------------------------------------
        _freqs = []
        lambda_list = []
        radii_list = []
        phase_list = []
        h_list = []
        k_list = []
        xy_list = []

        rad_pdeg_list = []
        ph_pdeg_list = []
        xc_pdeg_list = []
        yc_pdeg_list = []

        theta_fit = linspace(-pi, pi, 180)
        
        # -----------------------------------------------------------------------------------------------------------------
        if self._plot:
            # Initialize the plot figures
            fig = go.Figure()
            fig1 = go.Figure()
            fig2 = go.Figure()
            fig3 = go.Figure()
            # set the graph template in plotly
            large_rockwell_template = dict(
                layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
            
        # -----------------------------------------------------------------------------------------------------------------
        for idx, item in wrk_dict.items():
            # # extract the neighbours
            # info, xy_tuple = self.extract_neighbours(k)
            frequency_value = item[0]
            lambda_value = item[1]
            frq_name = f'Frequency = {frequency_value}'
            lambda_name = f'Lambda = {lambda_value}'
            log.debug(f'\n=> {frq_name} , {lambda_name}')

            self.automate_circle_extraction(idx, item[0], item[1])
            log.debug("Automation Finished !!! proceed further to evaluate model...")
            
            radius = self.res_dict.get('radius')
            (h, k) = self.res_dict.get('center')
            xy_tuple = self.res_dict.get('xy_tuple')
            residu = self.res_dict.get('residu')
            
            # Calculate the angle between center and the x,y coord of freq.
            # angle = atan2(y2-y1, x2-x1) where (x1,y1) is center and (x2, y2) is point on circle
            # ref : https://math.stackexchange.com/questions/94379/calculating-angle-in-circle
            angle_rad = math.atan2(xy_tuple[1] - k, xy_tuple[0] - h)
            if angle_rad < 0:
                angle_rad = (angle_rad + (2*math.pi))
                
            # log.debug("\n=> Angle in radians : {0}".format(angle_rad))
            log.debug("(lambda:{0}) => radius : {1}, angle : {2}, center(h : {3}, k : {4}), residu : {5}".format(
                lambda_value, radius, h, k, angle_rad, residu))
            log.debug("------------------------------------------------------")
            lambda_list.append(lambda_value)
            radii_list.append(radius)
            phase_list.append(angle_rad)
            h_list.append(h)
            k_list.append(k)
            xy_list.append(xy_tuple)  # contains list of xy tuples
            _freqs.append(frq)
            
            if self._plot:
                """
                Draw the data points, best fit circles and the center
                """
                f = p.figure(facecolor='white')
                p.axis('equal')
                ax = p.subplot(111)

                # plot the circles
                xc_2b = h
                yc_2b = k
                R_2b = radius

                x_fit2 = xc_2b + R_2b*cos(theta_fit)
                y_fit2 = yc_2b + R_2b*sin(theta_fit)

                ax.plot(x_fit2, y_fit2, 'k-', lw=1)

                # mark the center of the circle
                ax.plot([xc_2b], [yc_2b], 'gD', mec='r', mew=1)

                ax.plot([self._firstxy[0], h], [self._firstxy[1], k],
                        'm--', linewidth=0.7, label=f"Radius : {radius}")
                ax.plot([self._lastxy[0], h], [
                        self._lastxy[1], k], 'm--', linewidth=0.7)
                p.annotate("{:.2f}$^\circ$".format(
                    self.calc_angle), xy=(h, k+0.05))

                # draw
                p.xlabel('x')
                p.ylabel('y')

                # plot data points
                ax.plot(self.x, self.y, 'ro',
                        label=f'Neighbors: {self._neigh} , Step: {self._step}', ms=5, mec='b', mew=1)

                # Shrink current axis's height by 10% on the bottom
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.2,
                                box.width, box.height * 1.0])

                # Put a legend below current axis
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                          fancybox=True, shadow=True, ncol=5, labelspacing=0.1)
                ax.grid()
                p.title(
                    f'Circle Extraction (Least Squares Method) \n \n {frq_name}, {lambda_name}')
                # fmt_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                _frq_dir = PathUtils().create_freq_dir(frq)
                circ_plt = os.path.join(_frq_dir,  f'lsc_{frq}_{lambda_value}.png') # '{0}/' + f'lsc_{frq}_{lambda_value}.png'
                p_name = os.path.realpath(circ_plt)
                f.savefig(p_name, bbox_inches='tight', dpi=150)
                f.clf()
                f.clear()
                matplotlib.pyplot.close()
                
            log.info(f'{lambda_name}, Step : {self._step}, Neighbours : {self._neigh}')
            # reset the step and neighbors before using for processing the next freq.
            self._step = 4
            self._neigh = 10
            
            if self._plot:
                # Set plot axes properties
                fig.update_xaxes(zeroline=False)
                fig.update_yaxes(zeroline=False)

                # Scatter plot
                fig.add_trace(go.Scatter(x=[lambda_value], y=[radius],
                                         mode='markers',
                                         name=f'{lambda_name} , radius = {radius}'))

                fig1.add_trace(go.Scatter(
                    x=[lambda_value],
                    y=[angle_rad],
                    mode='markers',
                    name=f'{lambda_name} , angle = {angle_rad}'))

                fig2.add_trace(go.Scatter(
                    x=[lambda_value],
                    y=[h],
                    mode='markers',
                    name=f'{lambda_name} , h = {h}'))

                fig3.add_trace(go.Scatter(
                    x=[lambda_value],
                    y=[k],
                    mode='markers',
                    name=f'{lambda_name} , k = {k}'))
        # endfor
        
        if self._plot:
            # Line plots
            txt1 = ['Lambda : {}'.format(
                    lambda_list[k]) + ',<br>Radius : {}'.format(radii_list[k]) for k in range(len(lambda_list))]
            fig.add_trace(go.Scatter(
                x=lambda_list, y=radii_list, mode='lines', 
                text = txt1, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            txt2 = ['Lambda : {0}'.format(
                    lambda_list[k]) + ',<br>Phase : {0}'.format(phase_list[k]) for k in range(len(lambda_list))]
            fig1.add_trace(go.Scatter(
                x=lambda_list, y=phase_list, mode='lines', text = txt2, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            txt3 = ['Lambda : {0}'.format(
                    lambda_list[k]) + ',<br>X-Center : {0}'.format(h_list[k]) for k in range(len(lambda_list))]
            fig2.add_trace(go.Scatter(
                x=lambda_list, y=h_list, mode='lines', text = txt3, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            txt4 = ['Lambda : {0}'.format(
                    lambda_list[k]) + ',<br>X-Center : {0}'.format(k_list[k]) for k in range(len(lambda_list))]
            fig3.add_trace(go.Scatter(
                x=lambda_list, y=k_list, mode='lines', text = txt4, hoverinfo='text', hoverlabel=dict(namelength=-1)))

            # Set figure size
            fig.update_layout(title=f'Radius vs Lambda (Frequency = {frq})',
                              xaxis=dict(
                                  range=[7, 11],
                                  tickmode='linear',
                                  dtick=0.5),
                              yaxis=dict(
                                  range=[0.3, 1.0],
                                  tickmode='linear',
                                  dtick=0.1),
                              template=large_rockwell_template,
                              width=1000, height=600,
                              showlegend=True)

            # Change grid color and x and y axis colors
            fig.update_xaxes(gridcolor='black', griddash='dot')
            fig.update_yaxes(gridcolor='black', griddash='dot')

            plt_ht = os.path.join(self._plt_dir, f'Radius_Vs_Lambda_{frq}.html') # '{0}/' + f'Radius_Vs_Lambda_{frq}.html'
            fig.write_html(os.path.realpath(plt_ht))
            plt = os.path.join(self._plt_dir, f'Radius_Vs_Lambda_{frq}.png') # '{0}/' + f'Radius_Vs_Lambda_{frq}.png'
            f_name = os.path.realpath(plt)
            # fig.write_html(f_name)
            fig.write_image(f_name, engine="orca",
                            format="png", width=800, height=400, scale=2)

            # -----------------------------------------------------------------------------
            # Set figure1 size
            fig1.update_layout(title=f'Phase vs Lambda (Frequency = {frq})',
                               template=large_rockwell_template,
                               showlegend=True)
            fig1.update_xaxes(gridcolor='black', griddash='dot')
            fig1.update_yaxes(gridcolor='black', griddash='dot')
            plt1_ht = os.path.join(self._plt_dir, f'Phase_Vs_Lambda_{frq}.html') # '{0}/' + f'Phase_Vs_Lambda_{frq}.html'
            fig1.write_html(os.path.realpath(plt1_ht))
            plt1 = os.path.join(self._plt_dir, f'Phase_Vs_Lambda_{frq}.png') # '{0}/' + f'Phase_Vs_Lambda_{frq}.png'
            # fig1.write_html(os.path.realpath(plt1.format(self._plt_dir)))
            f1_name = os.path.realpath(plt1)
            fig1.write_image(f1_name, engine="orca",
                            format="png", width=800, height=400, scale=2)

            # -----------------------------------------------------------------------------
            # Set figure2 size
            fig2.update_layout(title=f'X-coord vs Lambda (Frequency = {frq})',
                               xaxis=dict(
                                   range=[7, 11],
                                   tickmode='linear',
                                   dtick=1.0),
                               yaxis=dict(
                                   range=[0.3, 0.6],
                                   tickmode='linear',
                                   dtick=0.1),
                               template=large_rockwell_template,
                               showlegend=True)
            fig2.update_xaxes(gridcolor='black', griddash='dot')
            fig2.update_yaxes(gridcolor='black', griddash='dot')
            plt2_ht = os.path.join(self._plt_dir, f'X-Coord_Vs_Lambda_{frq}.html') # '{0}/' + f'X-Coord_Vs_Lambda_{frq}.html'
            fig2.write_html(os.path.realpath(plt2_ht))
            plt2 = os.path.join(self._plt_dir, f'X-Coord_Vs_Lambda_{frq}.png') # '{0}/' + f'X-Coord_Vs_Lambda_{frq}.png'
            f2_name = os.path.realpath(plt2)
            fig2.write_image(f2_name, engine="orca",
                             format="png", width=800, height=400, scale=2)

            # -----------------------------------------------------------------------------
            # Set figure3 size
            fig3.update_layout(title=f'y-coord vs Lambda (Frequency = {frq})',
                               xaxis=dict(
                                   range=[7, 11],
                                   tickmode='linear',
                                   dtick=1.0),
                               yaxis=dict(
                                   range=[-0.4, -0.1],
                                   tickmode='linear',
                                   dtick=0.1),
                               template=large_rockwell_template,
                               showlegend=True)
            fig3.update_xaxes(gridcolor='black', griddash='dot')
            fig3.update_yaxes(gridcolor='black', griddash='dot')
            plt3_ht = os.path.join(self._plt_dir, f'Y-Coord_Vs_Lambda_{frq}.html') # '{0}/' + f'Y-Coord_Vs_Lambda_{frq}.html'
            fig3.write_html(os.path.realpath(plt3_ht))
            plt3 = os.path.join(self._plt_dir, f'Y-Coord_Vs_Lambda_{frq}.png') # '{0}/' + f'Y-Coord_Vs_Lambda_{frq}.png'
            f3_name = os.path.realpath(plt3)
            fig3.write_image(f3_name, engine="orca", 
                             format="png", width=800, height=400, scale=2)
            
        # check if the model score is good enough
        reject_model, deg_dict = self.evaluate_models(
            frq, lambda_list, phase_list, radii_list, h_list, k_list)
        # create dataframe with the desired attributes for regression modeling
        # {'radius_degree' : dm.radius_degree, 'phase_degree' : dm.phase_degree,
        # 'xc_degree' : dm.xcenter_degree, 'yc_degree' : dm.ycenter_degree
        rdeg = [deg_dict.get('radius_degree')]*len(_freqs)
        rad_pdeg_list = rdeg.copy()
        ph_deg = [deg_dict.get('phase_degree')]*len(_freqs)
        ph_pdeg_list = ph_deg.copy()
        xc_deg = [deg_dict.get('xc_degree')]*len(_freqs)
        xc_pdeg_list = xc_deg.copy()
        yc_deg = [deg_dict.get('yc_degree')]*len(_freqs)
        yc_pdeg_list = yc_deg.copy()

        _df = pd.DataFrame(list(zip(_freqs, lambda_list, radii_list, 
                                    phase_list, h_list, k_list, xy_list, 
                                    rad_pdeg_list, ph_pdeg_list, xc_pdeg_list, yc_pdeg_list)),
                           columns=['Frequency', 'Lambda', 'Radius', 'Angle', 
                                    'x_center', 'y_center', 'coordinates', 
                                    'radius_degree', 'phase_degree', 
                                    'xc_degree', 'yc_degree'])
        # _df = pd.DataFrame(list(zip(_freqs, lambda_list, radii_list, 
        #                             phase_list, h_list, k_list, xy_list)),
        #                    columns=['Frequency', 'Lambda', 'Radius', 'Angle', 
        #                             'x_center', 'y_center', 'coordinates'])
        
        # compute the maximum residue for the current freq. and print the associate lambda value
        # _tmp_res = zip(*self.residu_list)
        # _tmp_res = list(_tmp_res)

        # #log.info(lambda_list)

        # max_residu = max(_tmp_res[1]) #since 1st index contains the residu values
        # res_idx = _tmp_res[1].index(max_residu)
        # lval_res = _tmp_res[0][res_idx]
        # log.info(f"Maximum Residu : {max_residu}, Lambda : {lval_res}")

        self.residu_list.clear()
        #self._df_list.append(_df)

        if not reject_model:
            self._df_list.append(_df)
        return reject_model
    
     # -------------------------------------------------------------------------------------------------------------
    def process_circle_extraction(self):
        # create a copy of the dataset
        frf_df_cp = DataParser().get_freq_data()
        frf_df6 = frf_df_cp.copy(deep=True)

        # Pick the each row of these columns containing frequency, lambda, S1 real, S1 imaginary
        # and form a list of each row containing these column values.
        self.orgDF_list = frf_df6.apply(lambda row:
                                        [row['Frequency']]
                                        + [row['Lambda']]
                                        + [row['S1_Real[RE]']]
                                        + [row['S1_Imaginary[Im]']], axis=1).to_list()

        log.info(f'Length original dataframe : {len(self.orgDF_list)}')

        # ----------------------------------------------------------------------------------------------------------------

        # pick a random frequency from the data , 'n' denotes : number of samples to be picked
        # In this case we just pick one frequency and create a model for that.
        # df_elements = frf_df6.sample(n=3)
        # # contains the row of the picked frequency
        # log.info(f'<--------- Randomly Picked Row --------->\n{df_elements}')

        start_time = time.monotonic()
        log.info('------------------- START --------------------------')

        # ----------------------------------------------------------------------------------------------------------------

        # extract the freq. to a list
        # frq_list = df_elements['Frequency'].to_list()

        frq_list = frf_df6['Frequency'].to_list()
        _frqs = list(dict.fromkeys(frq_list))  # remove the duplicate freq.

        #log.info(frq_list)
        # log.info(f'Frequency List Size : {len(_frqs)}')
        # _frqs = [34.200000762939 , 34.209999084473 , 34.220001220703 , 34.229999542236 , 34.240001678467 ,
        #         34.25 , 34.259998321533 , 34.270000457764 , 34.279998779297 , 34.290000915527 , 34.299999237061 , 34.310001373291 ,
        #         34.319999694824 , 34.330001831055 , 34.340000152588 , 34.349998474121 , 34.360000610352 , 34.369998931885 , 34.380001068115 ,
        #         34.389999389648 , 34.400001525879 , 34.409999847412 , 34.419998168945 , 34.430000305176 , 34.439998626709 , 34.450000762939 ,
        #         34.459999084473]
        # _frqs = [34.200000762939, 34.209999084473]
        # _frqs = [34.310001373291]
        # _frqs = [25.0]
        #_frqs = [25.040000915527]
        log.info(f'Frequency List Size : {len(_frqs)}')

        for i, frq in enumerate(_frqs, start=1):
            # # reset the step and neighbors before using for processing the next freq.
            # self._step = 4
            # self._neigh = 10
            # extracted frequency.
            log.info(f'Picked Frequency : {frq} , Processing freq. {i}')
            # For the picked frequency extract all the matching rows
            if not frf_df6.empty:
                result_df = frf_df6[frf_df6['Frequency'].isin([frq])]
                result_df.to_csv(os.path.realpath(
                    os.path.join(PathUtils().get_data_directory(), 'extracted_frequency_df.csv')))
                # result_df.to_csv('extracted_frequency_df.csv')
            else:
                log.info('Dataframe is empty,cannot continue...!!')

            # (Convert to dictionary) working dict after the data extraction based on the picked frequency
            wrk_dict = result_df.apply(lambda row:
                                       [row['Frequency']]
                                       + [row['Lambda']]
                                       + [row['S1_Real[RE]']]
                                       + [row['S1_Imaginary[Im]']], axis=1).to_dict()
            log.debug("------------------------------------------------------")
            self.reject_model = self.evaluate_circle_extraction(frq, wrk_dict)
            # while (self.reject_model):
            #     if(self.ref_angle < MAX_REF_ANGLE):
            #         self.ref_angle += 2.
            #         log.info(f"Incremented Reference Angle : {self.ref_angle}")
            #     # increase the ref. angle between the first, last and center and process the extraction again!!!
            #     self.reject_model = self.evaluate_circle_extraction(frq, wrk_dict)

        end_time = time.monotonic()
        log.info(f'Finished in {format_timespan(end_time - start_time)}')
        log.info('-------------------- END ---------------------------')
        
    def _get_df_list(self):
        log.debug(f'LeastSquareCircle(): _get_df_list = {len(self._df_list)}')
        return self._df_list

    def dist_to_point(self):
        return math.sqrt((self.x1-self.x2)**2+(self.y1-self.y2)**2)
    
    def get_angle(self, a: tuple, b: tuple, c: tuple) -> np.degrees:
        # convert the tuples to numpy array
        a = np.array([*a])
        b = np.array([*b])
        c = np.array([*c])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / \
            (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)
    
     # check if the model score is good enough
    def evaluate_models(self, freq, lamdalst, phaselst, radiilst, hlst, klst):
        # for the given freq. iterate over the list lambda values
        # and prepare the model with polynomial degree 'x' if the
        # degree is good enough create a map of freq, with degree
        # otherwise increment the ploynomial degree, check if the model
        # score improves otherwise increment the ref. angle and extract
        # the circle parameters and evaluate the model again.
        dm = DataModeling()
        # dm.set_df_list(dframe)
        _loop_count = 5
        c1 = 0
        c2 = 0
        c3 = 0
        c4 = 0
        dm.fq = freq

        m1 = True
        m2 = True
        m3 = True
        m4 = True

        while (self.radii_mscore < REF_MODEL_SCORE and c1 < _loop_count):
            curr_radiiscore = dm.model_radius(lamdalst, radiilst)
            if (curr_radiiscore > self.radii_mscore):
                log.debug(
                    f'Radii MScore prev : {self.radii_mscore}, curr : {curr_radiiscore}')
                self.radii_mscore = curr_radiiscore
                dm.radius_degree += 1
            c1 += 1
            # model score is still bad reject the model and extract
            # the circle parameters by incrementing the ref. angle
            m1 = (ceil(self.radii_mscore*100) < (REF_MODEL_SCORE*100))
            if (m1 and c1 == _loop_count):
                m1 = False  # don't reject the model we can't improve the model score further here

        while (self.phase_mscore < REF_MODEL_SCORE and c2 < _loop_count):
            curr_phasescore = dm.model_phase(lamdalst, phaselst)
            if (curr_phasescore > self.phase_mscore):
                log.debug(
                    f'Phase MScore prev : {self.phase_mscore}, curr : {curr_phasescore}')
                self.phase_mscore = curr_phasescore
                dm.phase_degree += 1
            c2 += 1
            # model score is still bad reject the model and extract
            # the circle parameters by incrementing the ref. angle
            m2 = (ceil(self.phase_mscore*100) < (REF_MODEL_SCORE*100))
            if (m2 and c2 == _loop_count):
                m2 = False  # don't reject the model we can't improve the model score further here

        while (self.xc_mscore < REF_MODEL_SCORE and c3 < _loop_count):
            curr_xcscore = dm.model_xcenter(lamdalst, hlst)
            if (curr_xcscore > self.xc_mscore):
                log.debug(
                    f'Xc MScore prev : {self.xc_mscore}, curr : {curr_xcscore}')
                self.xc_mscore = curr_xcscore
                dm.xcenter_degree += 1
            c3 += 1
            # model score is still bad reject the model and extract
            # the circle parameters by incrementing the ref. angle
            m3 = (ceil(self.phase_mscore*100) < (REF_MODEL_SCORE*100))
            if (m3 and c3 == _loop_count):
                m3 = False  # don't reject the model we can't improve the model score further here

        # log.info(f"is Y score less then ref :  {self.yc_mscore < REF_MODEL_SCORE}")
        # log.info(f'curr count : count : {count}, loop count : {_loop_count}')
        # log.info(f'condition : {self.yc_mscore < REF_MODEL_SCORE and count < _loop_count}')
        while (self.yc_mscore < REF_MODEL_SCORE and c4 < _loop_count):
            curr_ycscore = dm.model_ycenter(lamdalst, klst)
            if (curr_ycscore > self.yc_mscore):
                log.debug(
                    f'Yc MScore prev : {self.yc_mscore}, curr : {curr_ycscore}')
                self.yc_mscore = curr_ycscore
                dm.ycenter_degree += 1
            c4 += 1
            # model score is still bad reject the model and extract
            # the circle parameters by incrementing the ref. angle
            m4 = (ceil(self.yc_mscore*100) < (REF_MODEL_SCORE*100))
            if (m4 and c4 == _loop_count):
                m4 = False  # don't reject the model we can't improve the model score further here

        log.info(f"m1 : {m1}, m2 : {m2}, m3 : {m3}, m4 : {m4}")
        reject_model = (m1 or m2 or m3 or m4)
        log.debug(
            f"PolynomialDegree of Models Radius: {dm.radius_degree}, Phase: {dm.phase_degree}, X-Center : {dm.xcenter_degree}, Y-Center : {dm.ycenter_degree}")
        log.info(
            f"Model Scores radius : {self.radii_mscore}, phase : {self.phase_mscore}, x-center : {self.xc_mscore}, y-center : {self.yc_mscore}")
        # log.info(f"Tried Stepwidth : {self._step}, Neighbours : {self._neigh}")
        log.info(f"Reject Model : {reject_model}")

        # clear the results
        self.radii_mscore = 0.
        self.phase_mscore = 0.
        self.xc_mscore = 0.
        self.yc_mscore = 0.
        dm.reset_polynomial_degree()
        return reject_model, {'radius_degree': dm.radius_degree,
                              'phase_degree': dm.phase_degree,
                              'xc_degree': dm.xcenter_degree,
                              'yc_degree': dm.ycenter_degree}

        # model are prepared time to predict
        # predict for three values of lambda to evaluate if
        # the model performance is good.

        # pred_lamlist = [lamdalst[0], lamdalst[len(lamdalst//2)], lamdalst[-1]]
        # for it in pred_lamlist:
        #     dm.predict_radius(it)
        #     dm.predict_phase(it)
        #     dm.predict_Xc(it)
        #     dm.predict_Yc(it)