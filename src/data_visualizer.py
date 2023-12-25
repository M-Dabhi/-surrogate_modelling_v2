import os
import random
import logging
import plotly.graph_objects as go

from pathlib import Path
from utils.path_utils import PathUtils
from data_parser import DataParser
from helper.singleton import Singleton

log = logging.getLogger(__name__)

class DataVisualizer(metaclass=Singleton):
    
    def __init__(self):
        self._res_path = PathUtils().get_results_directory()
        self.frf_df = DataParser().get_freq_data()    # get frequency response function data frame

    # def get_freq_dataframe(self):
    #     return self.frf_df
    
    def plot_frf_data(self):
        plt_name = os.path.realpath(os.path.join(self._res_path, 'frf_plot.html'))
        if Path(plt_name).is_file():
            log.info("Frequency Plot exists, skipping...")
        else:
            log.info("Plotting Frequency Curves...")
            large_rockwell_template = dict(
                layout=go.Layout(title_font=dict(family="Rockwell", size=24)))

            frf_df2 = self.frf_df.copy(deep=True)

            # Create traces
            fig = go.Figure()

            # extract the 'Lambda'values from dataframe
            lm_lst = frf_df2.apply(lambda row: row['Lambda'], axis=1).tolist()

            # remove the duplicate values from the list
            lm_lst = list(dict.fromkeys(lm_lst))

            for b in lm_lst:
                if not frf_df2.empty:
                    rslt_frf_df = frf_df2[frf_df2['Lambda'].isin([b])]
                    x_val = rslt_frf_df['Frequency'].to_list()
                    y_val = rslt_frf_df['Magnitude'].to_list()
                    lm_name = f'Lambda = {b}'
                    fig.add_trace(go.Scatter(x=x_val, y=y_val,
                                             mode='lines',
                                             name=lm_name))
                    rslt_lst = rslt_frf_df.index
                    frf_df2 = frf_df2.drop(rslt_lst, axis=0)
                else:
                    continue

            fig.update_layout(title='S-Parameters [Magnitude]',
                              xaxis=dict(
                                  tickmode='linear',
                                  tick0=25,
                                  dtick=1),
                              yaxis=dict(
                                  tickmode='linear',
                                  dtick=0.1),
                              template=large_rockwell_template)

            # Change grid color and axis colors
            fig.update_xaxes(gridcolor='black', griddash='dot')
            fig.update_yaxes(gridcolor='black', griddash='dot')
            fig.update_traces(line=dict(width=1))

            fig.write_html(plt_name)
            
    def draw_mag_parameter_plot(self):
        plt_name = os.path.realpath(os.path.join(self._res_path, 'lambda_mag_plot.html'))
        if Path(plt_name).is_file():
            log.info("Magnitude Vs S-Paramater plot exists, skipping...")
        else:
            log.info("Plotting Magnitude Vs S-Parameter...")
            # set the graph template in plotly
            large_rockwell_template = dict(
                layout=go.Layout(title_font=dict(family="Rockwell", size=24)))

            frf_df1 = self.frf_df.copy(deep=True)
            fq_lst = frf_df1.apply(
                lambda row: row['Frequency'], axis=1).tolist()

            # remove the duplicate values from the list
            # this is done to extract the values of lambda and magnitude at a given frequency
            fq_lst = list(dict.fromkeys(fq_lst))
            # Create traces
            fig = go.Figure()

            for i in range(25):
                f = random.choice(fq_lst)
                if not frf_df1.empty:
                    rslt_frf_df = frf_df1[frf_df1['Frequency'].isin([f])]

                    x_val = rslt_frf_df['Lambda'].to_list()
                    y_val = rslt_frf_df['Magnitude'].to_list()
                    fq_name = f'Frequency = {f}'

                    # Add the x and y values to trace
                    fig.add_trace(go.Scatter(x=x_val, y=y_val,
                                             mode='lines',
                                             name=fq_name))
                else:
                    continue

            fig.update_layout(
                xaxis=dict(
                    tickmode='linear',
                    tick0=25,
                    dtick=0.5
                ),
                yaxis=dict(
                    tickmode='linear',
                    dtick=0.1
                ),
                template=large_rockwell_template
            )
            # Change grid color and axis colors
            fig.update_xaxes(gridcolor='black', griddash='dot')
            fig.update_yaxes(gridcolor='black', griddash='dot')
            fig.update_traces(line=dict(width=1))
            # fig.show()
            fig.write_html(plt_name)   
            
    def draw_polar_plot(self): 
        plt_name = os.path.realpath(os.path.join(self._res_path, 'polar_plot_25.html'))
        if Path(plt_name).is_file():
            log.info("Polar Plot exists, skipping...")
        else:
            log.info("Plotting polar plot")
            frf_df3 = self.frf_df.copy(deep=True)
            frf_df3 = frf_df3.sort_values('Lambda')

            large_rockwell_template = dict(
                layout=go.Layout(title_font=dict(family="Rockwell", size=24)))

            # Create traces
            fig = go.Figure()

            # extract the 'Lambda' values from dataframe
            lm_lst = frf_df3.apply(lambda row: row['Lambda'], axis=1).tolist()
            # remove the duplicate lambda values
            lm_lst = list(dict.fromkeys(lm_lst))
            # log.info(len(lm_lst))
            #lm_lst = [8.7688442211055, 8.7889447236181, 8.8090452261307]

            for b in lm_lst:
                if not frf_df3.empty:
                    rslt_frf_df = frf_df3[frf_df3['Lambda'].isin([b])]
                    r_val = rslt_frf_df['Magnitude'].to_list()
                    theta_list = rslt_frf_df['Phase(Deg)'].to_list()

                    lab = f'Lambda = {b}'
                    fig.add_trace(go.Scatterpolar(
                        r=r_val,
                        theta=theta_list,
                        mode='markers',
                        name=lab))

                    rslt_lst = rslt_frf_df.index
                    frf_df3 = frf_df3.drop(rslt_lst, axis=0)
                else:
                    continue

            fig.update_layout(title=' S-Parameters [Polar Plot]',
                              polar=dict(
                                  radialaxis=dict(range=[0, 1], dtick=0.2),
                                  angularaxis=dict(
                                      thetaunit="degrees", dtick=30)
                              ), template=large_rockwell_template
                              )

            fig.update_xaxes(gridcolor='black', griddash='dot')
            fig.update_yaxes(gridcolor='black', griddash='dot')
            fig.update_traces(marker=dict(size=5))
            # save the plot to the local disk and open image from the disk
            fig.write_html(plt_name)
            
    
    def get_polar_plot_fig(self):
        frf_df4 = self.frf_df.copy(deep=True)
        frf_df4 = frf_df4.sort_values('Lambda')
        large_rockwell_template = dict(
            layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
        # Create traces
        fig = go.Figure()
        # extract the 'Lambda' values from dataframe
        lm_lst = frf_df4.apply(lambda row: row['Lambda'], axis=1).tolist()
        # remove the duplicate lambda values
        lm_lst = list(dict.fromkeys(lm_lst))
        #log.info(len(lm_lst))
        #lm_lst = [7.00000000000000]
        for b in lm_lst:
            if not frf_df4.empty:
                rslt_frf_df = frf_df4[(frf_df4['Lambda'].isin([b]))]
                #rslt_frf_df = frf_df4[(frf_df4['Lambda'].isin([b])) & 
                #                      (frf_df4['Lambda'] > 10.3) & 
                #                      (frf_df4['Lambda'] < 10.7) &
                #                      (frf_df4['Frequency'] >= 25.0) &
                #                      (frf_df4['Frequency'] <= 25.11)]
                #log.info(len(rslt_frf_df))
                r_val = rslt_frf_df['Magnitude'].to_list()
                theta_list = rslt_frf_df['Phase(Deg)'].to_list()
                lab = f'Lambda = {b}'
                fig.add_trace(go.Scatterpolar(
                    r=r_val,
                    theta=theta_list,
                    mode='markers',
                    name=lab))
                rslt_lst = rslt_frf_df.index
                frf_df4 = frf_df4.drop(rslt_lst, axis=0)
            else:
                continue
            
        fig.update_traces(
            hoverlabel=dict(
                namelength=-1  # Set to -1 for unlimited length, adjust as needed
            )
        )
        
        fig.update_layout(title=' S-Parameters [Lambda Polar Plot]',
                          polar=dict(
                              radialaxis=dict(range=[0, 1], dtick=0.2),
                              #angularaxis=dict(
                              #    thetaunit="degrees", dtick=30)
                          ), template=large_rockwell_template
                          )
        fig.update_xaxes(gridcolor='black', griddash='dot')
        fig.update_yaxes(gridcolor='black', griddash='dot')
        fig.update_traces(marker=dict(size=5))
        return fig
    
    
    def get_polar_plot_fig_wrt_freq(self):
        plt_name = os.path.realpath(os.path.join(self._res_path, 'polar_plot_lambda(7.0).html'))
        frf_df4 = self.frf_df.copy(deep=True)
        large_rockwell_template = dict(
            layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
        # Create traces
        fig = go.Figure()
        # extract the 'Frequency' values from dataframe
        freq_lst = frf_df4.apply(lambda row: row['Frequency'], axis=1).tolist()
        # remove the duplicate lambda values
        freq_lst = list(dict.fromkeys(freq_lst))
        #log.info(len(lm_lst))
        for b in freq_lst:
            if not frf_df4.empty:
                rslt_frf_df = frf_df4[(frf_df4['Frequency'].isin([b]))]
                #rslt_frf_df = frf_df4[(frf_df4['Frequency'].isin([b])) & (frf_df4['Lambda'] > 10.3) & (frf_df4['Lambda'] < 10.7)]
                #log.info(len(rslt_frf_df))
                r_val = rslt_frf_df['Magnitude'].to_list()
                theta_list = rslt_frf_df['Phase(Deg)'].to_list()
                lab = f'Frequency = {b}'
                fig.add_trace(go.Scatterpolar(
                    r=r_val,
                    theta=theta_list,
                    mode='markers',
                    name=lab))
                rslt_lst = rslt_frf_df.index
                frf_df4 = frf_df4.drop(rslt_lst, axis=0)
            else:
                continue
            
        fig.update_traces(
            hoverlabel=dict(
                namelength=-1  # Set to -1 for unlimited length, adjust as needed
            )
        )
        
        fig.update_layout(title=' S-Parameters [Frequency Polar Plot]',
                          polar=dict(
                              radialaxis=dict(range=[0, 1], dtick=0.2),
                              angularaxis=dict(
                                  thetaunit="degrees", dtick=30)
                          ), template=large_rockwell_template
                          )
        fig.update_xaxes(gridcolor='black', griddash='dot')
        fig.update_yaxes(gridcolor='black', griddash='dot')
        fig.update_traces(marker=dict(size=5))
        # save the plot to the local disk and open image from the disk
        #fig.write_html(plt_name)
        return fig
    
    def get_freq_vs_mag_fig(self):
        log.info('Ploting Frequency vs Magnitude Plot')
        large_rockwell_template = dict(
            layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
        frf_df2 = self.frf_df.copy(deep=True)
        # Create traces
        fig = go.Figure()
        # extract the 'Lambda'values from dataframe
        lm_lst = frf_df2.apply(lambda row: row['Lambda'], axis=1).tolist()
        # remove the duplicate values from the list
        lm_lst = list(dict.fromkeys(lm_lst))
        #log.info(f'Lambda list from Freq Vs Mag plot => {lm_lst}')
        for b in lm_lst:
            if not frf_df2.empty:
                rslt_frf_df = frf_df2[(frf_df2['Lambda'].isin([b]))]
                x_val = rslt_frf_df['Frequency'].to_list()
                #log.info(f'Freqency values from Freq vs Mag plot => {x_val}')
                y_val = rslt_frf_df['Magnitude'].to_list()
                lm_name = f'Lambda = {b}'
                #log.info(f'Lambda Value -> {lm_name}')
                fig.add_trace(go.Scatter(x=x_val, y=y_val,
                                         mode='markers',
                                         name=lm_name))
                rslt_lst = rslt_frf_df.index
                #log.info(f'index of rslt_lst : {rslt_lst}')
                frf_df2 = frf_df2.drop(rslt_lst, axis=0)
                #log.info(f'Result after drop : {frf_df2}')
            else:
                continue
            
        fig.update_traces(
            hoverlabel=dict(
                namelength=-1  # Set to -1 for unlimited length, adjust as needed
            )
        )
        
        fig.update_layout(title='S-Parameters [Frequency Vs Magnitude]',
                          xaxis=dict(
                                range=[24.9, 26],
                                tickmode='linear',
                                dtick=0.1),
                          yaxis=dict(
                              range=[0.01, 0.25],
                              tickmode='linear',
                              dtick=0.01),
                          template=large_rockwell_template)
        # Change grid color and axis colors
        fig.update_xaxes(gridcolor='black', griddash='dot')
        fig.update_yaxes(gridcolor='black', griddash='dot')
        fig.update_traces(line=dict(width=1))
        return fig
    
    
    def get_lambda_vs_mag_fig(self):
        log.info('Ploting lambda vs Magnitude Plot')
        # set the graph template in plotly
        large_rockwell_template = dict(
            layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
        frf_df1 = self.frf_df.copy(deep=True)
        fq_lst = frf_df1.apply(lambda row: row['Frequency'], axis=1).tolist()
        # remove the duplicate values from the list
        # this is done to extract the values of lambda and magnitude at a given frequency
        fq_lst = list(dict.fromkeys(fq_lst))
        # Create traces
        fig = go.Figure()
        for f in fq_lst:
            if not frf_df1.empty:
                rslt_frf_df = frf_df1[(frf_df1['Frequency'].isin([f])) & 
                                      (frf_df1['Lambda'] > 10.3) & (frf_df1['Lambda'] < 10.7)]
                x_val = rslt_frf_df['Lambda'].to_list()
                #log.info(f'Lambda Values from lambda_vs_mag plot => {x_val}')
                y_val = rslt_frf_df['Magnitude'].to_list()
                fq_name = f'Frequency = {f}'
                # Add the x and y values to trace
                fig.add_trace(go.Scatter(x=x_val, y=y_val,
                                         mode='markers',
                                         name=fq_name))
            else:
                continue
            
        fig.update_traces(
            hoverlabel=dict(
                namelength=-1  # Set to -1 for unlimited length, adjust as needed
            )
        )
        
        fig.update_layout(title='S-Parameters [Lambda Vs Magnitude]',
            xaxis=dict(
                range=[6.95, 11],
                tickmode='linear',
                dtick=0.5
            ),
            yaxis=dict(
                tickmode='linear',
                dtick=0.1
            ),
            template=large_rockwell_template
        )
        # Change grid color and axis colors
        fig.update_xaxes(gridcolor='black', griddash='dot')
        fig.update_yaxes(gridcolor='black', griddash='dot')
        fig.update_traces(line=dict(width=1))
        return fig

    