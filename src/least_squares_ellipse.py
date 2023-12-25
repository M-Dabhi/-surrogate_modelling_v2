import os
import numpy as np
import plotly.graph_objects as go
import plotly
import logging
import numpy_indexed as npi
import matplotlib

from scipy.optimize import minimize
from utils.path_utils import PathUtils
from data_parser import DataParser
from utils.configs import Configs
from matplotlib import pyplot as p
from matplotlib.patches import Ellipse
from data_visualizer import DataVisualizer

log = logging.getLogger(__name__)


class LeastSquaresEllipse:
    def __init__(self) -> None:
        self._neigh = 30
        self._step = 18
        self._plot = Configs()._draw_plots()
        self._firstxy = 0.
        self._lastxy = 0.
        plotly.io.orca.config.executable = os.path.realpath(os.path.join(PathUtils().get_current_directory(), 'orca_app', 'orca.exe')) # r'C:\ProgramData\miniconda3\orca_app\orca.exe'
        plotly.io.orca.config.save()
        self._res_path = PathUtils().get_results_directory()
        self.result_dict = {'semi-major axis' : 0., 'semi-minor axis' : 0., 'center' : (0.,0.)}
        self.mag_phase_dict = {'mag' : [], 'phase' : [], 'freq' : 0., 'lambda' : 0.}
        
    def ellipse_objective(self, params, *args):
        x, y = args
        a, b, h, k = params
        return np.sum((((x - h) / a)**2 + ((y - k) / b)**2 - 1)**2)
    
    def initialize_params(self, x_data, y_data):
        h_initial = np.mean(x_data)
        k_initial = np.mean(y_data)

        a_initial = np.sqrt(np.mean((x_data - h_initial)**2))
        b_initial = np.sqrt(np.mean((y_data - k_initial)**2))

        # Initial orientation angle is set to zero
        #theta_initial = 0

        return [a_initial, b_initial, h_initial, k_initial]
    
    def process_ellipse_extraction(self):
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
        _frqs = [30.0]
        log.info(f'Frequency List Size : {len(_frqs)}')
        
        for i, frq in enumerate(_frqs, start=1):
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
            
            #print(f"Working dictionary for freq {frq} : {wrk_dict}")
            self.evaluate_ellipse_extraction(frq, wrk_dict)
            
    
    def evaluate_ellipse_extraction(self, frq, wrk_dict):
        """
        - For each 'lambda' value in the working dataframe, we pick the index and 
            extract the adjacent data points for the picked index, which inturn contains 
            the adjacent frequencies data.
        - Pass the extracted real and imaginarly values as tuple to plot the ellipse 
            which passes through these 3 points.
        - With ellipse plot we also have ellipse 'Two axis' and 'Center' information.
        """
        _freqs = []
        lambda_list = []
        mag_phase_lst_dict = []
        semi_major_axis_list = []
        semi_minor_axis_list = []
        h_list = []
        k_list = []
        
        theta = np.linspace(0, 2*np.pi, 10000)
        
        if not self._plot:
            # Initialize the plot figures
            fig = go.Figure()
            fig1 = go.Figure()
            fig2 = go.Figure()
            fig3 = go.Figure()
            # set the graph template in plotly
            large_rockwell_template = dict(
                layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
            
        for idx, item in wrk_dict.items():
            frequency_value = item[0]
            lambda_value = item[1]
            frq_name = f'Frequency = {frequency_value}'
            lambda_name = f'Lambda = {lambda_value}'
            #log.info(f'\n=> {frq_name} , {lambda_name}')
            
            self.automate_ellipse_extraction(idx, frequency_value, lambda_value)
            
            a = self.result_dict.get('semi-major axis') # semi_major_axis
            b = self.result_dict.get('semi-minor axis') # semi_minor_axis
            (h, k) = self.result_dict.get('center')
            
            _freqs.append(frq)
            lambda_list.append(lambda_value)
            mag_phase_lst_dict.append(self.mag_phase_dict)
            semi_major_axis_list.append(a)
            semi_minor_axis_list.append(b)
            h_list.append(h)
            k_list.append(k)
            
            if not self._plot:
                """
                Draw the data points, best fit ellipse and the center
                """
                
                # Parametric equations for the ellipse
                x = h + a * np.cos(theta)
                y = k + b * np.sin(theta)
                ellipse_fig = p.figure(facecolor='white')
                #p.axis('tight')
                ax = p.subplot(111)
                ax.plot(x, y, 'k-', lw=1)
                
                # mark the center of the circle
                ax.plot([h], [k], 'gD', mec='r', mew=1)
                ax.plot([self._firstxy[0], h], [self._firstxy[1], k],
                        'm--', linewidth=0.7)
                ax.plot([self._lastxy[0], h], [
                        self._lastxy[1], k], 'm--', linewidth=0.7)
                
                # Draw lines for the semi-axes
                ax.plot([h - a, h + a], [k, k], 'b-', lw=1, label = f'Semi-major axis : {a}')  # Semi-major axis
                ax.plot([h, h], [k - b, k + b], 'g-', lw=1, label = f'Semi-major axis : {b}')  # Semi-minor axis

                p.xlabel('x')
                p.ylabel('y')
                
                # plot data points
                #log.info('X and Y coordinates : {0} and {1}'.format(self.x, self.y))
                ax.plot(self.x, self.y, 'ro',
                        label=f'Neighbors: {self._neigh} , Step: {self._step}', ms=5, mec='b', mew=1)
                
                # Put a legend below current axis
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                          fancybox=True, shadow=True, ncol=5, labelspacing=0.1)
                ax.grid()
                title=f'Ellipse Extraction (Least Squares Ellipse) \n \n {frq_name}, {lambda_name}'
                p.title(title)
                #p.tight_layout(rect=[0, 10, 1, 1])
                #ax.set_aspect('equal')
                
                # Set axis ticks
                #ax.set_xticks(np.arange(h-a,h+a,5))
                #ax.set_yticks(np.arange(h-b,k+b,5))

                # Set the size of the figure
                ellipse_fig.set_size_inches(14, 8)  # Adjust the size as needed
                
                #p.show()
                
                ellipse_plt = os.path.join(PathUtils().get_ellipse_plots_directory(), f'lse_{frq}_{lambda_value}.png')
                ellipse_fig.savefig(os.path.realpath(ellipse_plt), dpi=150)
                ellipse_fig.clf()
                ellipse_fig.clear()
                matplotlib.pyplot.close()
                
                #ellipse_fig = go.Figure()
                #ellipse_fig.add_trace(go.Scatter(x = x, y = y, mode="lines",
                #              fill='toself', fillcolor='rgba(0,100,80,0.2)', 
                #              line=dict(color='rgba(255,255,255,0)')))
                #ellipse_fig.update_layout(
                #    title=f'Ellipse Extraction (Least Squares Ellipse) {frq_name}, {lambda_name}',
                #    xaxis=dict(title="X-axis"),
                #    yaxis=dict(title="Y-axis"),
                #    showlegend=True
                #)
                #results_dir = PathUtils().get_results_directory()
                #ellipse_fig.write_html(os.path.realpath(os.path.join(results_dir, f'Ellipse_{frq_name}_{lambda_name}.html')))
                
            # end If
                
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
                    y=[h],
                    mode='markers',
                    name=f'{lambda_name} , h = {h}'))

                fig3.add_trace(go.Scatter(
                    x=[lambda_value],
                    y=[k],
                    mode='markers',
                    name=f'{lambda_name} , k = {k}'))
                
            # end if
            
            #break
        # end for
        
        _dv = DataVisualizer()
        pol_fig = _dv.get_polar_plot_fig()
        
        for itm in mag_phase_lst_dict:
            mag_values = itm['mag']
            phase_values = itm['phase']
            freq_value = itm['freq']
            lambda_value = itm['lambda']
            pol_fig.add_trace(go.Scatterpolar(
                r=mag_values,
                theta=phase_values*180/np.pi,
                mode='lines',
                name=f'Ellipse {freq_value} & {lambda_value}'
                ))
            
         # end for
         
        pol_fig.write_html(os.path.realpath(os.path.join(self._res_path, 'polar_plot_25_w_ellipse.html')))    
        
        if not self._plot:
            # Line plots
            txt1 = ['Lambda : {}'.format(lambda_list[k]) + 
                    ',<br> a : {}'.format(semi_major_axis_list[k]) for k in range(len(lambda_list))]
            fig.add_trace(go.Scatter(
                x=lambda_list, y=semi_major_axis_list, mode='lines', 
                text = txt1, hoverinfo='text', hoverlabel=dict(namelength=-1)))    
            
            txt2 = ['Lambda : {}'.format(lambda_list[k]) + 
                    ',<br> b : {}'.format(semi_minor_axis_list[k]) for k in range(len(lambda_list))]
            fig1.add_trace(go.Scatter(
                x=lambda_list, y=semi_minor_axis_list, mode='lines', 
                text = txt2, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            txt3 = ['Lambda : {}'.format(lambda_list[k]) + 
                    ',<br> X-Center : {}'.format(h_list[k]) for k in range(len(lambda_list))]
            fig2.add_trace(go.Scatter(
                x=lambda_list, y=h_list, mode='lines', 
                text = txt3, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            txt4 = ['Lambda : {}'.format(lambda_list[k]) + 
                    ',<br> Y-Center : {}'.format(k_list[k]) for k in range(len(lambda_list))]
            fig3.add_trace(go.Scatter(
                x=lambda_list, y=k_list, mode='lines', 
                text = txt4, hoverinfo='text', hoverlabel=dict(namelength=-1)))
            
            # Set figure size
            fig.update_layout(title=f'Semi Major Axis vs Lambda (Frequency = {frq})',
                              xaxis=dict(
                                  range=[7, 11],
                                  tickmode='linear',
                                  dtick=0.5),
                              yaxis=dict(
                                  range=[0, 60],
                                  tickmode='linear'
                                  ),
                              template=large_rockwell_template,
                              width=1500, height=900,
                              showlegend=True)
            # Change grid color and x and y axis colors
            fig.update_xaxes(gridcolor='black', griddash='dot')
            fig.update_yaxes(gridcolor='black', griddash='dot')
            
            frq_dir = PathUtils().create_freq_dir_for_plots(frq)
            plt_ht = os.path.join(frq_dir, f'Semi-Major-Axis(a)_Vs_Lambda_{frq}.html')
            fig.write_html(os.path.realpath(plt_ht))
            plt = os.path.join(frq_dir, f'Semi-Major-Axis(a)_Vs_Lambda_{frq}.png')
            f_name = os.path.realpath(plt)
            fig.write_image(f_name, engine="orca",
                            format="png", width=800, height=400, scale=2)
            
            # -----------------------------------------------------------------------------
            # Set figure1 size
            fig1.update_layout(title=f'Semi Minor Axis vs Lambda (Frequency = {frq})',
                                xaxis=dict(
                                  range=[7, 11],
                                  tickmode='linear',
                                  dtick=0.5),
                                yaxis=dict(
                                  range=[0, 20],
                                  tickmode='linear',
                                  dtick=0.5),
                               template=large_rockwell_template,
                               width=1500, height=900,
                               showlegend=True)
            fig1.update_xaxes(gridcolor='black', griddash='dot')
            fig1.update_yaxes(gridcolor='black', griddash='dot')
            
            plt1_ht = os.path.join(frq_dir, f'Semi-Minor-Axis_Vs_Lambda_{frq}.html')
            fig1.write_html(os.path.realpath(plt1_ht))
            plt1 = os.path.join(frq_dir, f'Semi-Minor-Axis_Vs_Lambda_{frq}.png')
            f1_name = os.path.realpath(plt1)
            fig1.write_image(f1_name, engine="orca",
                            format="png", width=800, height=400, scale=2)
            
             # -----------------------------------------------------------------------------
            # Set figure2 size
            fig2.update_layout(title=f'X-coord vs Lambda (Frequency = {frq})',
                               xaxis=dict(
                                   range=[7, 11],
                                   tickmode='linear',
                                   dtick=0.5),
                               yaxis=dict(
                                   range=[-40, 20],
                                   tickmode='linear'
                                   ),
                               template=large_rockwell_template,
                               width=1500, height=900,
                               showlegend=True)
            fig2.update_xaxes(gridcolor='black', griddash='dot')
            fig2.update_yaxes(gridcolor='black', griddash='dot')
            plt2_ht = os.path.join(frq_dir, f'X-Coord_Vs_Lambda_Ellipse_{frq}.html')
            fig2.write_html(os.path.realpath(plt2_ht))
            plt2 = os.path.join(frq_dir, f'X-Coord_Vs_Lambda_Ellispe_{frq}.png') 
            f2_name = os.path.realpath(plt2)
            fig2.write_image(f2_name, engine="orca",
                             format="png", width=800, height=400, scale=2)

            # -----------------------------------------------------------------------------
            # Set figure3 size
            fig3.update_layout(title=f'Y-coord vs Lambda (Frequency = {frq})',
                               xaxis=dict(
                                   range=[7, 11],
                                   tickmode='linear',
                                   dtick=0.5),
                               yaxis=dict(
                                   range=[-25, 10],
                                   tickmode='linear'
                                   ),
                               template=large_rockwell_template,
                               width=1500, height=900,
                               showlegend=True)
            fig3.update_xaxes(gridcolor='black', griddash='dot')
            fig3.update_yaxes(gridcolor='black', griddash='dot')
            plt3_ht = os.path.join(frq_dir, f'Y-Coord_Vs_Lambda_Ellipse_{frq}.html') 
            fig3.write_html(os.path.realpath(plt3_ht))
            plt3 = os.path.join(frq_dir, f'Y-Coord_Vs_Lambda_Ellipse_{frq}.png') 
            f3_name = os.path.realpath(plt3)
            fig3.write_image(f3_name, engine="orca", 
                             format="png", width=800, height=400, scale=2)

            
            
    def automate_ellipse_extraction(self, idx: int, freq: float, lmbda: float) -> dict:
        # 1. adjust the neighbours
        neigh_info, xy_tuple = self.adjust_neighbours_frqs(idx, freq, lmbda)
        
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
        
        initial_params = self.initialize_params(self.x, self.y)
        
        result = minimize(self.ellipse_objective, initial_params, args=(self.x, self.y), 
                          method = 'SLSQP')
        
        a, b, h, k = result.x
        #log.info('Best fitting ellipse parameters : {0}'.format(result))
        
        def ellipse(theta, x_center, y_center, a, b):
            return np.sqrt((a * np.cos(theta))**2 + (b * np.sin(theta))**2) + np.sqrt(x_center**2 + y_center**2)
        
        theta_values = np.linspace(0, 2*np.pi, 100)
        
        # Calculate points on the ellipse
        r_values = ellipse(theta_values, h, k, a, b)
        
        #log.info(f'mag : {r_values} & phase : {theta_values}')
        
    
        #log.info('a : {0}, b : {1}, h : {2}, k : {3}'.format(a,b,h,k))
        self.result_dict = {'semi-major axis' : a, 'semi-minor axis' : b, 'center' : (h, k)}
        self.mag_phase_dict = {'mag' : r_values, 'phase' : theta_values, 'freq' : freq, 'lambda' : lmbda}
        #log.info('Result Dictionary ---------> {0}'.format(self.result_dict))            
        
        
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
        print(f'\nPicked lambda -->  {pickd_lm}, Picked Frq ---> {pickd_frq}')
        
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