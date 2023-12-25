import os
import math
import natsort
import numpy as np
from numpy import *
import sympy as sp
import plotly.graph_objects as go
import plotly
import logging
import numpy_indexed as npi
import matplotlib

from pathlib import Path
from scipy.optimize import curve_fit
from utils.path_utils import PathUtils
from data_parser import DataParser
from utils.configs import Configs
from matplotlib import pyplot as p
from matplotlib.patches import Circle
from plotly.offline import plot
from data_visualizer import DataVisualizer
from scipy.interpolate import CubicSpline as cs

log = logging.getLogger(__name__)

MAX_NEIGH = 25
MAX_STEP_WIDTH = 25
MAX_REF_ANGLE = 21.

class CubicSpline:
    def __init__(self) -> None:
        self._neigh = 30
        self._step = 4
        self._plot = Configs()._draw_plots()
        self._firstxy = 0.
        self._lastxy = 0.
        self.calc_angle = 0.
        self.ref_angle = 19.
        self.ref_angle_thresholds = [self.ref_angle - 0.5, self.ref_angle + 2]
        plotly.io.orca.config.executable = os.path.realpath(os.path.join(PathUtils().get_current_directory(), 'orca_app', 'orca.exe')) # r'C:\ProgramData\miniconda3\orca_app\orca.exe'
        plotly.io.orca.config.save()
        self._res_path = PathUtils().get_results_directory()
        self.result_dict = {'coeff a' : 0., 
                            'coeff b' : 0., 
                            'coeff c' : 0., 
                            'coeff d' : 0.}
        self.mag_phase_dict = {'mag' : [], 'phase' : [], 'freq' : 0., 'lambda' : 0.}
        self.circle_res_dict = {'radius' : 0., 'center' : (0., 0.), 'tangent' : 0., 'xy_tuple' : (0., 0.)}
        self.spline_dict = {'x_spline' : [], 'y_spline' : []}
    
        self.adjusted = False
        self.is_larger = False
        self.is_smaller = False
        
    # Define the cubic spline function
    def cubic_spline(self, x, a, b, c, d):
        x = np.array(x)
        return a + b * x + c * x**2 + d * x**3
    
    def process_spline_extraction(self):
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
        
        frq_list = frf_df6['Frequency'].to_list()
        _frqs = [25.049999237061, 30.0]
        #_frqs = [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0]
        log.info(f'Frequency List Size : {len(_frqs)}')
        
        for i, frq in enumerate(_frqs, start=1):
            log.info(f'Picked Frequency : {frq} , Processing freq. {i}')
            # For the picked frequency extract all the matching rows
            if not frf_df6.empty:
                result_df = frf_df6[(frf_df6['Frequency'].isin([frq]))]
                #result_df = frf_df6[(frf_df6['Frequency'].isin([frq])) & (frf_df6['Lambda'] > 10.3) & (frf_df6['Lambda'] < 10.7)]
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
            
            #print(f"Working dictionary for freq {frq} : {wrk_dict}")
            self.evaluate_spline_extraction(frq, wrk_dict)
            
            
    def evaluate_spline_extraction(self, frq, wrk_dict):
        _freqs = []
        lambda_list = []
        coeff_a_lst = []
        coeff_b_lst = []
        coeff_c_lst = []
        coeff_d_lst = []
        h_list = []
        k_list = []
        radius_list = []
        xy_list = []
        phase_list = []
        mag_phase_lst_dict = []
        
        r_cicle_lst = []
        theta_circle_lst = []
        
        theta = np.linspace(0, 2*np.pi, 10000)
        
        theta_fit = np.linspace(-np.pi, np.pi, 180)
        
        if not self._plot:
            # Initialize the plot figures
            fig = go.Figure()
            fig1 = go.Figure()
            fig2 = go.Figure()
            fig3 = go.Figure()
            fig4 = go.Figure()
            fig5 = go.Figure()
            fig6 = go.Figure()
            fig7 = go.Figure()
            # set the graph template in plotly
            large_rockwell_template = dict(
                layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
            
        for idx, item in wrk_dict.items():
            frequency_value = item[0]
            lambda_value = item[1]
            frq_name = f'Frequency = {frequency_value}'
            lambda_name = f'Lambda = {lambda_value}'
            #log.info(f'\n=> {frq_name} , {lambda_name}')
            
            self.automate_spline_extraction(idx, frequency_value, lambda_value)
            
            a = self.result_dict.get('coeff a')
            b = self.result_dict.get('coeff b')
            c = self.result_dict.get('coeff c')
            d = self.result_dict.get('coeff d')
            radius = self.circle_res_dict.get('radius')
            (h, k) = self.circle_res_dict.get('center')
            xy_tuple = self.circle_res_dict.get('xy_tuple')
            
            angle_rad = math.atan2(xy_tuple[1] - k, xy_tuple[0] - h)
            if angle_rad < 0:
                angle_rad = (angle_rad + (2*math.pi))
            
            _freqs.append(frq)
            lambda_list.append(lambda_value)
            coeff_a_lst.append(a)
            coeff_b_lst.append(b)
            coeff_c_lst.append(c)
            coeff_d_lst.append(d)
            h_list.append(h)
            k_list.append(k)
            radius_list.append(radius) 
            phase_list.append(angle_rad)
            mag_phase_lst_dict.append(self.mag_phase_dict)
            
            # Convert center to polar coordinates
            r_center = np.sqrt(h**2 + k**2)
            theta_center = np.arctan2(k, h)
            # Generate points on the circle
            angles = np.linspace(0, 2*np.pi, 100)
            r = r_center + radius * np.cos(angles)
            #theta = theta_center + radius * np.sin(angles)
            theta = np.mod(angles + theta_center, 2*np.pi)
            r_points = np.zeros_like(angles)
            theta_points = np.zeros_like(angles)
            
            circle_x = h + radius * np.cos(angles)
            circle_y = k + radius * np.sin(angles)
            circle_r = np.sqrt(circle_x**2, circle_y**2)
            circle_theta = np.arctan2(circle_y, circle_x)
            # Calculate the polar coordinates of the circle's points
            for i, angle in enumerate(angles):
                # Calculate the Cartesian coordinates of the point on the circle
                x = h + radius * np.cos(angle)
                y = k + radius * np.sin(angle)

                # Convert the point's Cartesian coordinates to polar coordinates
                r_points[i] = np.sqrt(x**2 + y**2)
                theta_points[i] = np.arctan2(y, x)
                
            r_cicle_lst.append(r_points)
            theta_circle_lst.append(theta_points)
            
            log.info(f'{lambda_name}, Step : {self._step}, Neighbours : {self._neigh}')
            # reset the step and neighbors before using for processing the next freq.
            #self._step = 4
            #self._neigh = 10
            
            if self._plot:
                f = p.figure(facecolor='white')
                p.axis('equal')
                ax = p.subplot(111)

                # plot the circles
                xc_2b = h
                yc_2b = k
                R_2b = radius

                x_fit2 = xc_2b + R_2b*cos(theta_fit)
                y_fit2 = yc_2b + R_2b*sin(theta_fit)

                x_spline = self.spline_dict.get('x_spline')
                y_spline = self.spline_dict.get('y_spline')
                tangent = self.circle_res_dict.get('tangent')

                ax.plot(x_fit2, y_fit2, 'k-', lw=1)

                # mark the center of the circle
                ax.plot([xc_2b], [yc_2b], 'gD', mec='r', mew=1)

                # plot data points
                ax.plot(self.x, self.y, 'ro')

                # draw
                p.xlabel('x')
                p.ylabel('y')

                #plot Spline
                ax.plot(x_spline, y_spline, label = 'Cubic Spline')

                tangent_end_x = x_spline[0] + 0.1
                tangent_end_y = y_spline[0] + 0.1 * tangent

                #plot tangent
                ax.plot([x_spline[0], tangent_end_x], [y_spline[0], tangent_end_y], 'r-')
                ax.set_aspect('equal', adjustable='box')
                ax.legend()
                ax.grid()
                p.title(
                        f'Circle Extraction (Cubic Spline) \n \n {frq_name}, {lambda_name}')
                _frq_dir = PathUtils().get_spline_plots_directory(frq)
                circle_plt = os.path.join(_frq_dir, f'Spline_with_circle_{frq}_{lambda_value}.png')
                f.savefig(os.path.realpath(circle_plt))
                f.clf()
                f.clear()
                matplotlib.pyplot.close()
                 
            
            if not self._plot:
                # Set plot axes properties
                fig.update_xaxes(zeroline=False)
                fig.update_yaxes(zeroline=False)
                
                # Scatter plot
                fig.add_trace(go.Scatter(
                    x=[lambda_value], 
                    y=[a],
                    mode='markers',
                    name=f'{lambda_name} , a = {a}'))
                
                fig1.add_trace(go.Scatter(
                    x=[lambda_value],
                    y=[b],
                    mode='markers',
                    name=f'{lambda_name} , b = {b}'))

                fig2.add_trace(go.Scatter(
                    x=[lambda_value],
                    y=[c],
                    mode='markers',
                    name=f'{lambda_name} , c = {c}'))
                
                fig3.add_trace(go.Scatter(
                    x=[lambda_value],
                    y=[d],
                    mode='markers',
                    name=f'{lambda_name} , d = {d}'))
                
                fig4.add_trace(go.Scatter(
                    x=[lambda_value],
                    y=[h],
                    mode='markers',
                    name=f'{lambda_name} , h = {h}'))
                
                fig5.add_trace(go.Scatter(
                    x=[lambda_value],
                    y=[k],
                    mode='markers',
                    name=f'{lambda_name} , k = {k}'))
                
                fig6.add_trace(go.Scatter(
                    x=[lambda_value],
                    y=[radius],
                    mode='markers',
                    name=f'{lambda_name} , radius = {radius}'))
                
                fig7.add_trace(go.Scatter(
                    x=[lambda_value],
                    y=[angle_rad],
                    mode='markers',
                    name=f'{lambda_name} , angle = {angle_rad}'))
            
            #break
            
        #end for
        
        if not self._plot:
            _dv = DataVisualizer()
            pol_fig_lmdba = _dv.get_polar_plot_fig()
            pol_fig_freq = _dv.get_polar_plot_fig_wrt_freq()
            #lambda_vs_mag_fig = _dv.get_lambda_vs_mag_fig()
            #freq_vs_mag_fig = _dv.get_freq_vs_mag_fig()
            #
            for itm in mag_phase_lst_dict:
                mag_values = itm['mag']
                phase_values = itm['phase']
                freq_value = itm['freq']
                lambda_value = itm['lambda']
                pol_fig_lmdba.add_trace(go.Scatterpolar(
                    r=mag_values,
                    theta=phase_values*180/np.pi,
                    mode='lines',
                    name=f'(Cubic Spline) {freq_value} & {lambda_value}'
                    ))
                pol_fig_freq.add_trace(go.Scatterpolar(
                    r=mag_values,
                    theta=phase_values*180/np.pi,
                    mode='lines',
                    name=f'(Cubic Spline) {freq_value} & {lambda_value}'
                    ))
            #    freq_vs_mag_fig.add_trace(go.Scatter(
            #        x = _freqs,
            #        y = mag_values,
            #        mode = 'lines',
            #        name = f'Cubic Spline {freq_value} & {lambda_value}'
            #    ))
            #    lambda_vs_mag_fig.add_trace(go.Scatter(
            #        x = lambda_list,
            #        y = mag_values,
            #        mode = 'lines',
            #        name = f'Cubic Spline {freq_value} & {lambda_value}'
            #    ))

            # end for

            circle_fig, ax1 = p.subplots(subplot_kw={'projection':'polar'})
            figg = go.Figure() 
            #log.info(f'r list : {r_cicle_lst} and theta list : {theta_circle_lst}')
            for r, theta, lambda_value  in zip(r_cicle_lst, theta_circle_lst, lambda_list):
                pol_fig_lmdba.add_trace(go.Scatterpolar(
                    r=r,
                    theta=theta*180/np.pi,
                    mode='lines',
                    name=f'(Circle) {lambda_value}'
                    ))
                pol_fig_freq.add_trace(go.Scatterpolar(
                    r=r,
                    theta=theta*180/np.pi,
                    mode='lines',
                    name=f'(Circle) {lambda_value}'
                    ))
                figg.add_trace(go.Scatterpolar(
                    r=r,
                    theta=theta*180/np.pi,
                    mode='lines',
                    name=f'(Circle) {lambda_value}'
                    ))
                ax1.plot(theta, r)
            
            figg.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=False
            )
            
            figg.write_html(os.path.realpath(os.path.join(self._res_path, f'polar_plot_{frq}_only_circles_1.html')))
            # end for
            circle_fig.savefig(os.path.realpath(os.path.join(PathUtils().get_results_directory(), f'circles_{frq}_3.png')))
            #
            pol_fig_lmdba.write_html(os.path.realpath(os.path.join(self._res_path, f'polar_plot_{frq}_wrt_lambda_cs.html')))
            pol_fig_freq.write_html(os.path.realpath(os.path.join(self._res_path, f'polar_plot_{frq}_wrt_freq_cs.html')))
            #freq_vs_mag_fig.write_html(os.path.realpath(os.path.join(self._res_path, f'freq_vs_mag_{frq}_cs.html')))
            #lambda_vs_mag_fig.write_html(os.path.realpath(os.path.join(self._res_path, f'lambda_vs_mag_{frq}_cs.html')))
        
        if not self._plot:
            # Line plots
            txt = ['Lambda : {}'.format(lambda_list[k]) + 
                    ',<br> a : {}'.format(coeff_a_lst[k]) for k in range(len(lambda_list))]
            fig.add_trace(go.Scatter(
                x=lambda_list, y=coeff_a_lst, mode='lines', 
                text = txt, hoverinfo='text', hoverlabel=dict(namelength=-1))) 
            
            txt1 = ['Lambda : {}'.format(lambda_list[k]) + 
                    ',<br> b : {}'.format(coeff_b_lst[k]) for k in range(len(lambda_list))]
            fig1.add_trace(go.Scatter(
                x=lambda_list, y=coeff_b_lst, mode='lines', 
                text = txt1, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            txt2 = ['Lambda : {}'.format(lambda_list[k]) + 
                    ',<br> c : {}'.format(coeff_c_lst[k]) for k in range(len(lambda_list))]
            fig2.add_trace(go.Scatter(
                x=lambda_list, y=coeff_c_lst, mode='lines', 
                text = txt2, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            txt3 = ['Lambda : {}'.format(lambda_list[k]) + 
                    ',<br> d : {}'.format(coeff_d_lst[k]) for k in range(len(lambda_list))]
            fig3.add_trace(go.Scatter(
                x=lambda_list, y=coeff_d_lst, mode='lines', 
                text = txt3, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            txt4 = ['Lambda : {}'.format(
                    h_list[k]) + ',<br>X-Center : {}'.format(h_list[k]) for k in range(len(lambda_list))]
            fig4.add_trace(go.Scatter(
                x=lambda_list, y=h_list, mode='lines', 
                text = txt4, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            txt5 = ['Lambda : {}'.format(
                    h_list[k]) + ',<br>Y-Center : {}'.format(k_list[k]) for k in range(len(lambda_list))]
            fig5.add_trace(go.Scatter(
                x=lambda_list, y=k_list, mode='lines', 
                text = txt5, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            txt6 = ['Lambda : {}'.format(
                    h_list[k]) + ',<br>Radius : {}'.format(radius_list[k]) for k in range(len(lambda_list))]
            fig6.add_trace(go.Scatter(
                x=lambda_list, y=radius_list, mode='lines', 
                text = txt6, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            txt7 = ['Lambda : {0}'.format(
                    lambda_list[k]) + ',<br>Phase : {0}'.format(phase_list[k]) for k in range(len(lambda_list))]
            fig7.add_trace(go.Scatter(
                x=lambda_list, y=phase_list, mode='lines', 
                text = txt7, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            #log.info(coeff_a_lst)
            # Set figure size
            fig.update_layout(title=f' Cubic Spline Coeff(a) vs Lambda (Frequency = {frq})',
                              xaxis=dict(
                                  range=[7, 11],
                                  tickmode='linear',
                                  dtick=0.5),
                              yaxis=dict(
                                  range=[min(coeff_a_lst)-0.1, max(coeff_a_lst)+0.1],
                                  tickmode='linear',
                                  dtick=max(coeff_a_lst)-min(coeff_a_lst)/len(lambda_list)*1.5),
                              template=large_rockwell_template,
                              width=1500, height=900,
                              showlegend=True)
            # Change grid color and x and y axis colors
            fig.update_xaxes(gridcolor='black', griddash='dot')
            fig.update_yaxes(gridcolor='black', griddash='dot')
            
            frq_dir = PathUtils().create_freq_dir_for_plots(frq)
            plt_ht = os.path.join(frq_dir, f'Cubic_Spline_Coeff(a)_Vs_Lambda_{frq}.html')
            fig.write_html(os.path.realpath(plt_ht))
            plt = os.path.join(frq_dir, f'Cubic_Spline_Coeff(a)_Vs_Lambda_{frq}.png')
            f_name = os.path.realpath(plt)
            fig.write_image(f_name, engine="orca",                                
                            format="png", width=800, height=400, scale=2)
        
        
            # -----------------------------------------------------------------------------
            # Set figure1 size
            fig1.update_layout(title=f'Cubic Spline Coeff(b) vs Lambda (Frequency = {frq})',
                                xaxis=dict(
                                  range=[7, 11],
                                  tickmode='linear',
                                  dtick=0.5),
                                yaxis=dict(
                                  range=[-4, 15],
                                  tickmode='linear',
                                  dtick=0.5),
                               template=large_rockwell_template,
                               width=1500, height=900,
                               showlegend=True)
            fig1.update_xaxes(gridcolor='black', griddash='dot')
            fig1.update_yaxes(gridcolor='black', griddash='dot')
            plt1_ht = os.path.join(frq_dir, f'Cubic_Spline_Coeff(b)_Vs_Lambda_{frq}.html')
            fig1.write_html(os.path.realpath(plt1_ht))
            plt1 = os.path.join(frq_dir, f'Cubic_Spline_Coeff(b)_Vs_Lambda_{frq}.png')
            f1_name = os.path.realpath(plt1)
            fig1.write_image(f1_name, engine="orca",
                            format="png", width=800, height=400, scale=2)
            
            # -----------------------------------------------------------------------------
            # Set figure2 size
            fig2.update_layout(title=f'Cubic Spline Coeff(c) vs Lambda (Frequency = {frq})',
                               xaxis=dict(
                                   range=[7, 11],
                                   tickmode='linear',
                                   dtick=0.5),
                               yaxis=dict(
                                   range=[-54, 54],
                                   tickmode='linear',
                                   dtick=2
                                   ),
                               template=large_rockwell_template,
                               width=1500, height=900,
                               showlegend=True)
            fig2.update_xaxes(gridcolor='black', griddash='dot')
            fig2.update_yaxes(gridcolor='black', griddash='dot')
            plt2_ht = os.path.join(frq_dir, f'Cubic_Spline_Coeff(c)_Vs_Lambda_{frq}.html') 
            fig2.write_html(os.path.realpath(plt2_ht))
            plt2 = os.path.join(frq_dir, f'Cubic_Spline_Coeff(c)_Vs_Lambda_{frq}.png') 
            f2_name = os.path.realpath(plt2)
            fig2.write_image(f2_name, engine="orca",
                             format="png", width=800, height=400, scale=2)
            
              # -----------------------------------------------------------------------------
            # Set figure3 size
            fig3.update_layout(title=f'Cubic Spline Coeff(d) vs Lambda (Frequency = {frq})',
                               xaxis=dict(
                                   range=[7, 11],
                                   tickmode='linear',
                                   dtick=0.5),
                               yaxis=dict(
                                   range=[-50, 50],
                                   tickmode='linear',
                                   dtick=2),
                               template=large_rockwell_template,
                               width=1500, height=900,
                               showlegend=True)
            fig3.update_xaxes(gridcolor='black', griddash='dot')
            fig3.update_yaxes(gridcolor='black', griddash='dot')
            plt3_ht = os.path.join(frq_dir, f'Cubic_Spline_Coeff(d)_Vs_Lambda_{frq}.html') 
            fig3.write_html(os.path.realpath(plt3_ht))
            plt3 = os.path.join(frq_dir, f'Cubic_Spline_Coeff(d)_Vs_Lambda_{frq}.png') 
            f3_name = os.path.realpath(plt3)
            fig3.write_image(f3_name, engine="orca", 
                             format="png", width=800, height=400, scale=2)
            
                # -----------------------------------------------------------------------------
            # Set figure4 size
            fig4.update_layout(title=f'X-coord vs Lambda (Frequency = {frq})',
                              xaxis=dict(
                                  range=[7, 11],
                                  tickmode='linear',
                                  dtick=0.5),
                              yaxis=dict(
                                  range=[min(h_list), max(h_list)],
                                  tickmode='linear',
                                  dtick=0.005),
                              template=large_rockwell_template,
                              width=1500, height=900,
                              showlegend=True)

            # Change grid color and x and y axis colors
            fig4.update_xaxes(gridcolor='black', griddash='dot')
            fig4.update_yaxes(gridcolor='black', griddash='dot')

            plt4_ht = os.path.join(frq_dir, f'X-Coord_Vs_Lambda_CS_{frq}.html') 
            fig4.write_html(os.path.realpath(plt4_ht))
            plt4 = os.path.join(frq_dir, f'Radius_Vs_Lambda_CS_{frq}.png')
            f_name = os.path.realpath(plt4)
            fig4.write_image(f_name, engine="orca",
                            format="png", width=800, height=400, scale=2)
            
            # -----------------------------------------------------------------------------
            # Set figure5 size
            fig5.update_layout(title=f'Y-coord vs Lambda (Frequency = {frq})',
                               xaxis=dict(
                                   range=[7, 11],
                                   tickmode='linear',
                                   dtick=0.5),
                               yaxis=dict(
                                   range=[min(k_list), max(k_list)],
                                   tickmode='linear',
                                   dtick=0.005),
                               template=large_rockwell_template,
                               width=1500, height=900,
                               showlegend=True)
            fig5.update_xaxes(gridcolor='black', griddash='dot')
            fig5.update_yaxes(gridcolor='black', griddash='dot')
            plt5_ht = os.path.join(frq_dir, f'Y-Coord_Vs_Lambda_CS_{frq}.html') 
            fig5.write_html(os.path.realpath(plt5_ht))
            plt5 = os.path.join(frq_dir, f'Y-Coord_Vs_Lambda_CS_{frq}.png')
            f_name = os.path.realpath(plt5)
            fig3.write_image(f_name, engine="orca", 
                             format="png", width=800, height=400, scale=2)
            
            # -----------------------------------------------------------------------------
            # Set figure6 size
            fig6.update_layout(title=f'Radius vs Lambda (Frequency = {frq})',
                              xaxis=dict(
                                  range=[7, 11],
                                  tickmode='linear',
                                  dtick=0.5),
                              yaxis=dict(
                                  range=[min(radius_list), max(radius_list)],
                                  tickmode='linear',
                                  dtick=0.005),
                              template=large_rockwell_template,
                              width=1500, height=900,
                              showlegend=True)

            # Change grid color and x and y axis colors
            fig6.update_xaxes(gridcolor='black', griddash='dot')
            fig6.update_yaxes(gridcolor='black', griddash='dot')

            plt6_ht = os.path.join(frq_dir, f'Radius_Vs_Lambda_CS_{frq}.html') 
            fig6.write_html(os.path.realpath(plt6_ht))
            plt6 = os.path.join(frq_dir, f'Radius_Vs_Lambda_CS_{frq}.png')
            f_name = os.path.realpath(plt6)
            fig.write_image(f_name, engine="orca",
                            format="png", width=800, height=400, scale=2)
            
            # -----------------------------------------------------------------------------
            # Set figure1 size
            fig7.update_layout(title=f'Phase vs Lambda (Frequency = {frq})',
                                xaxis=dict(
                                  range=[7, 11],
                                  tickmode='linear',
                                  dtick=0.5),
                                yaxis=dict(
                                  range=[2.5, 6],
                                  tickmode='linear',
                                  dtick=0.1),
                               template=large_rockwell_template,
                               width=1500, height=900,
                               showlegend=True)
            fig7.update_xaxes(gridcolor='black', griddash='dot')
            fig7.update_yaxes(gridcolor='black', griddash='dot')
            plt7_ht = os.path.join(frq_dir, f'Phase_Vs_Lambda_CS_{frq}.html')
            fig7.write_html(os.path.realpath(plt7_ht))
            plt7 = os.path.join(frq_dir, f'Phase_Vs_Lambda_CS_{frq}.png') 
            f_name = os.path.realpath(plt7)
            fig7.write_image(f_name, engine="orca",
                            format="png", width=800, height=400, scale=2)

        
    
    def automate_spline_extraction(self, idx: int, freq: float, lmbda: float) -> dict:
        # 1. adjust the neighbours
        neigh_info, xy_tuple = self.adjust_neighbours_frqs(idx, freq, lmbda)
        #log.info('neighbours before sorting : {0}'.format(neigh_info))
        
        #neigh_info = natsort.natsorted(neigh_info, key= lambda x:x[2])
        #log.info('neighbours after sorting : {0}'.format(neigh_info))
        
        # extract the coordinate tuples from the neighbour info
        self.x = []
        self.y = []
        neigh_freq = []
        for item in neigh_info:
            # 0 : Freq and 1 : Lambda from 2: Coordinates
            co_ord = tuple(item[2:])
            # extract the first element x of the tuple into list
            self.x.append(co_ord[0])
            # extract the second element of the tuple into a list
            self.y.append(co_ord[1])
            neigh_freq.append(item[0])
            
        log.debug(
            "\nExtracted X-Points : {0} \
            \n Extracted Y-Points : {1} \
            \n Adj. Neighbour Frequencies : {2}".format(self.x, self.y, neigh_freq))
        
        # Initial guesses for coefficients a, b, c, and d
        initial_guess = (0, 0, 0, 0)
        
        #Fit the cubic spline using least squares minimization
        params, covariance = curve_fit(self.cubic_spline, self.x, self.y, p0=initial_guess)

        # Extract the optimized coefficients
        a_opt, b_opt, c_opt, d_opt = params
        
        # Generate points for the cubic spline
        x_spline = self.x
        y_spline = self.cubic_spline(x_spline, a_opt, b_opt, c_opt, d_opt)
        
        #h, k, radius, tangent = self.approximate_circle_params(self.x, a_opt, b_opt, c_opt, d_opt, freq)
        
        h, k, radius = self.create_tangent_for_freq(neigh_freq, freq, self.x, self.y, y_spline, lmbda)
        
        cn_lst = [complex(x, y) for x, y in zip(x_spline, y_spline)]
        r_lst = [abs(cn) for cn in cn_lst]
        phase_lst = [np.angle(cn, deg=True) for cn in cn_lst]
        
        r_lst = np.sqrt(np.array(x_spline)**2 + np.array(y_spline)**2)
        phase_lst = np.arctan2(y_spline, x_spline)
        
        self.calc_angle = self.get_angle(self._firstxy, (h, k), self._lastxy)
        
        #if self.ref_angle_thresholds[0] <= self.calc_angle <= self.ref_angle_thresholds[1]:
        #    self.adjusted = True
        #    self.is_smaller = False
        #    self.is_larger = False
        #elif self.calc_angle < self.ref_angle_thresholds[0]:
        #    self.is_smaller = True
        #    self.is_larger = False
        #    self.adjusted = False
        #elif self.calc_angle > self.ref_angle_thresholds[1]:
        #    self.is_larger = True
        #    self.is_smaller = False
        #    self.adjusted = False
        #    
        #if (self.adjusted == True):
        #    pass
        #else:
        #    if self.is_smaller:
        #        while (self.is_smaller):
        #            if (self._step < MAX_STEP_WIDTH):
        #                self._step += 1
        #            if (self._step >= 10 and self._neigh < MAX_NEIGH):
        #                self._neigh += 1
        #            self.automate_spline_extraction(idx, freq, lmbda)  # update res_dict
        #            # update is_smaller based on the updated value of calc_angle
        #            self.is_smaller = self.calc_angle < self.ref_angle_thresholds[0]
        #            if self.ref_angle_thresholds[0] <= self.calc_angle <= self.ref_angle_thresholds[1]:
        #                self.adjusted = True
        #        # end while angle has been adjusted
        #    elif self.is_larger:
        #        while (self.is_larger):
        #            if (self._step < MAX_STEP_WIDTH):
        #                self._step -= 1
        #            if (self._step >= 10 and self._neigh < MAX_NEIGH):
        #                self._neigh += 1
        #            self.automate_spline_extraction(idx, freq, lmbda)  # update res_dict
        #            # update is_larger based on the updated value of calc_angle
        #            self.is_larger = self.calc_angle > self.ref_angle_thresholds[1]
        #            if self.ref_angle_thresholds[0] <= self.calc_angle <= self.ref_angle_thresholds[1]:
        #                self.adjusted = True
        
        #add trace to existing polat plot
        
        # Read the existing polar plot
        #_dv = DataVisualizer()
        
        #fig = _dv.get_polar_plot_fig()
        #fig.add_trace(go.Scatterpolar(
        #    r=r_lst,
        #    theta=phase_lst,
        #    mode='lines',
        #    name=f'Optimization Result {freq}'
        #))
    
        #fig.write_html(os.path.realpath(os.path.join(self._res_path, 'polar_plot_25_w_cs.html')))

        
        #log.info('Params & Covariance : {0} & {1}'.format(params, covariance))
        self.result_dict = {'coeff a' : a_opt, 'coeff b' : b_opt, 'coeff c' : c_opt, 'coeff d' : d_opt}
        self.mag_phase_dict = {'mag' : r_lst, 'phase' : phase_lst, 'freq' : freq, 'lambda' : lmbda}
        self.circle_res_dict = {'radius' : radius, 'center' : (h, k), 'tangent' : 0., 'xy_tuple' : xy_tuple}
        self.spline_dict = {'x_spline' : x_spline, 'y_spline' : y_spline}
        #log.info('Result Dictionary ---------> {0}'.format(self.result_dict))  
        
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
        
        
    def create_tangent_for_freq(self, neigh_freq, freq, x_coord, y_coord, y_spline, lambda_val):
        #log.info(f'Neighbours info => {neigh_freq}')
        #log.info(f'Length of Neighbours => {len(neigh_freq)}')
        #log.info(f'X Coordinate => {x_coord}')
        #log.info(f'Y Coordinate => {y_coord}')
        freq_idx = neigh_freq.index(freq)
        #log.info(f'Index of {freq} => {freq_idx}')
        total_neigh = len(neigh_freq)
        start = len(neigh_freq[:freq_idx])
        stop = len(neigh_freq[freq_idx+1:])
        t = np.linspace(-start, stop, total_neigh)
        #log.info(f't values => {t}')
        x_t = cs(t, x_coord)
        y_t = cs(t, y_coord)
        
        dx_dt = x_t.derivative()
        dy_dt = y_t.derivative()
        
        ddx_dt = dx_dt.derivative()
        ddy_dt = dy_dt.derivative()
        
        tangent = np.array([dx_dt(0), dy_dt(0)])
        #tangent = np.array([dx_dt(0), dy_dt(0)]) / np.sqrt(dx_dt(0)**2 + dy_dt(0)**2)
        #log.info(f'tangent at 0 => {tangent}')
        
        numerator = dx_dt(0) * ddy_dt(0) - dy_dt(0) * ddx_dt(0)
        denominator = (dx_dt(0)**2 + dy_dt(0)**2)**1.5
        curvature = numerator / denominator
        
        #dT_t = np.array([ddx_dt(0), ddy_dt(0)]) / np.sqrt(dx_dt(0)**2 + dy_dt(0)**2)
        #ds_t = np.array([dx_dt(0), dy_dt(0)])
        #k = dT_t/ds_t
        #curvature = np.sqrt(k[0]**2 + k[1]**2)
        
        #radius = 1 / curvature
        
        #unit_normal_vector = np.array([-ddy_dt(0), ddx_dt(0)]) / np.sqrt(dx_dt(0)**2 + dy_dt(0)**2)
        #center = np.array([x_t(0), y_t(0)]) + radius * unit_normal_vector
        
        radius = np.abs(((dx_dt(0)**2 + dy_dt(0)**2)**1.5)/(dx_dt(0)*ddy_dt(0) - dy_dt(0)*ddx_dt(0)))
        
        T_unit = tangent / np.linalg.norm(tangent)
        N = np.array([-T_unit[1], T_unit[0]])
        C = np.array([x_t(0), y_t(0)]) + np.sign(curvature) * radius * N
        
        #log.info(f'Curvature at 0 => {curvature}')
        #log.info(f'Unit tangent vector => {T_unit}')
        #log.info(f'Radius at 0 => {radius}')
        #log.info(f'Center at 0 => {C}')
        
        #x_vals = np.array([x_t(v) for v in t])
        #y_vals = np.array([y_t(v) for v in t])
                
        f = p.figure(facecolor='white')
        ax = p.subplot(111)
        ax.plot(x_coord, y_coord, 'bo')
        #ax.plot(x_vals, y_vals, 'yo')
        ax.plot(x_coord, y_spline, label='Cubic Spline')
        ax.plot([C[0]], [C[1]], 'gD', mec='r', mew = 1)
        ax.plot([x_coord[freq_idx], C[0]], [y_coord[freq_idx], C[1]], 'm--')
        ax.grid()
        circle = Circle(C, radius, fill = False)
        p.gca().add_patch(circle)
        p.quiver(x_coord[freq_idx], y_coord[freq_idx], dx_dt(0)*5, dy_dt(0)*5, color='red', scale=3, scale_units='xy', angles='xy')
        p.xlabel('x')
        p.ylabel('y')
        p.legend()
        p.title(f'Circle Extraction (Cubic Spline) \n \n {freq}, {lambda_val}')
        _frq_dir = PathUtils().get_spline_plots_directory(freq)
        circle_plt = os.path.join(_frq_dir, f'Spline_with_circle_{freq}_{lambda_val}.png')
        f.savefig(os.path.realpath(circle_plt))
        f.clf()
        f.clear()
        matplotlib.pyplot.close()
        
        return C[0], C[1], radius
        
        
    def approximate_circle_params(self, x, a_val, b_val, c_val, d_val, freq):
        # Symbolic variables
        x_sym, a, b, c, d = sp.symbols('x a b c d', real=True)
        y = a + b * (x_sym - x[0]) + c * (x_sym - x[0])**2 + d * (x_sym - x[0])**3
        
        # First derivative (tangent vector)
        dy_dx = sp.diff(y, x_sym)
        T = sp.Matrix([1, dy_dx])
        
        # Unit tangent vector
        U = T / T.norm()

        # Normal vector
        N = sp.Matrix([[0, -1], [1, 0]]) @ U

        # Second derivative (for curvature)
        d2y_dx2 = sp.diff(dy_dx, x_sym)
        T_prime = sp.Matrix([1, d2y_dx2])

        # Curvature
        curvature = T_prime.norm() / T.norm()
        
        # Circle center and radius
        circle_center = sp.Matrix([x_sym, y]) + 1 / curvature * N
        circle_radius = 1 / curvature
        
        # Convert to numerical functions
        circle_center_func = sp.lambdify((x_sym, a, b, c, d), circle_center, 'numpy')
        circle_radius_func = sp.lambdify((x_sym, a, b, c, d), circle_radius, 'numpy')
        tangent_func = sp.lambdify((x_sym, a, b, c, d), dy_dx, 'numpy')
        
        center = circle_center_func(x[0], a_val, b_val, c_val, d_val)
        radius = circle_radius_func(x[0], a_val, b_val, c_val, d_val)
        tangent = tangent_func(x[0], a_val, b_val, c_val, d_val)
        
        h = center[0][0]
        k = center[1][0]

        #log.info(f'Circle center : {center} and radius : {radius} and tangent : {tangent} for frequency : {freq}')
        #log.info(f'h : {center[0][0]} and k = {center[1][0]}')
        
        return h, k, radius, tangent
        
        #fig, ax = plt.subplots()
        #
        #y_spline = self.cubic_spline(x, a_val, b_val, c_val, d_val)
        #ax.plot(x, y_spline, label='Cubic Spline')
        #tangent_end_x = x[0] + 0.1
        #tangent_end_y = y_spline[0] + 0.1 * tangent
        #ax.plot([x[0], tangent_end_x], [y_spline[0], tangent_end_y], 'r-')
        #
        #circle = plt.Circle((center[0][0], center[1][0]), radius, color = 'y', fill = False)
        #ax.add_patch(circle)
        #
        #ax.set_aspect('equal', adjustable='box')
        #ax.set_title(f'Frequency : {freq}')
        #ax.legend()
        #plt.show()
        
        

    def adjust_neighbours_frqs(self, idx, freq, lmbda):
        # extract the neighbours
        info, xy_tuple = self.extract_neighbours(idx)
        frequency_value = freq
        lambda_value = lmbda
        frq_name = f'Frequency = {frequency_value}'
        lambda_name = f'Lambda = {lambda_value}'
        log.debug('=> Extracted Info : {0} \n=> (x,y) Tuple : {1}'.format(info, xy_tuple))
        log.debug(f'\n=> {frq_name} , {lambda_name}')
        
        # calculate the distance between the first and last x,y points
        self._firstxy, self._lastxy = tuple(info[0][-2:]), tuple(info[-1][-2:])
        log.debug(f"firstxy : {self._firstxy}, lastxy : {self._lastxy}")
        
        return info, xy_tuple
        
    def extract_neighbours(self, idx):
        orgDFs = self.orgDF_list[:]  # copy list into new variable
        _pick = orgDFs[idx] 
        extract_xy = _pick[-2:]  # type list
        xy_tuple = tuple(extract_xy)
        log.debug('=> Picked Row using index {0} from dataframe : {1}'.format(idx, _pick))
        log.debug('=> Extracted x,y tuple : {0}'.format(xy_tuple))
        
        # copy list into new variable so we don't change it
        main_list = orgDFs[:]
        _res_list = self.pick_every_x_element(main_list, idx)
        #log.info('List after picking elements : {0}'.format(_res_list))
        
        # update the index with the new working list
        _new_idx = _res_list.index(_pick)
        log.debug('New index from Picked Element List : {0}'.format(_new_idx))
        log.debug(f'extract_neighbours: Neigh Count --> {self._neigh}')
        left_slice = _new_idx - self._neigh//2  # start of range
        
        left_slice = min(max(0, left_slice), len(_res_list) - self._neigh)
        log.debug('left_slice # Start of range ----> {0}'.format(left_slice))
        
        # extract the right slice range
        right_slice = left_slice + self._neigh  # end of range
        log.debug('right_slice # End of range ----> {0}'.format(right_slice))
        
        log.debug('Extracted Neighbours ----------> {0}'.format(_res_list[left_slice:right_slice]))
        
        return _res_list[left_slice:right_slice], xy_tuple
        
        
    def pick_every_x_element(self, lst, index):
        pickd_lm = lst[index][1]
        pickd_frq = lst[index][0]
        #print(f'\nPicked lambda -->  {pickd_lm}, Picked Frq ---> {pickd_frq}')
        
        filtered_data = list(filter(lambda x: x[1] == pickd_lm, lst))
        filtered_data_array = np.array(filtered_data)
        searched_values = np.array([lst[index]])
        log.debug('Searched Value: {0}'.format(searched_values))
        _result = npi.indices(filtered_data_array, searched_values)
        log.debug('Resulted Value: {0}'.format(_result))
        
        if _result is not None:
            index = _result[-1]
        else:
            index = -1
            
        # Check if the given index is valid
        if index < 0 or index >= len(filtered_data):
            return None
        
        log.debug(f'pick_every_x_element: Step Count --> {self._step}')
        # Extract every 1th element from the sublist, starting from the given index
        r_sublist = filtered_data[index::self._step]
        #log.info('r_sublist ------------> {0}'.format(r_sublist))
        # Extract every 1th element to the left sublist, starting from the given index
        sub_slice = filtered_data[:index+1]
        #log.info('sub_slice ------------------> {0}'.format(sub_slice))
        l_sublist = sub_slice[::-self._step]
        #log.info('l_sublist ------------> {0}'.format(l_sublist))
        l_sublist = list(reversed(l_sublist))
        # as picked index is added twice remove it from the left sublist and join with the right sublist
        l_sublist.remove(l_sublist[-1])
        _f_list = l_sublist + r_sublist
        
        #log.info('Final list -------------> {0}'.format(_f_list))
        
        return _f_list
    