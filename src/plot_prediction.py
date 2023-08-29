import os
import pandas as pd
import logging
import plotly.graph_objects as go

from mse import MeanSquaredError
from utils.path_utils import PathUtils

class PlotPrediction:
    def __init__(self, _final_df) -> None:
        self._final_df = _final_df
        self.predict_mag_plot()
        # calculate the mean square error between the original and predicted curves.
        MeanSquaredError(_final_df)
        
    def predict_mag_plot(self):
        frf_df3 = self._final_df.copy(deep=True)
        df_elements = frf_df3.sample(n=20)

        picked_lm = df_elements['Lambda'].to_list()
        picked_lm.append(7.0)
        picked_lm.append(11.0)
        picked_lm = list(dict.fromkeys(picked_lm))
        _res_df = frf_df3[frf_df3['Lambda'].isin(picked_lm)]
        _res_df.reset_index(drop=True, inplace=True)
        
        # Create traces
        fig = go.Figure()

        # extract the 'Lambda'values from dataframe
        lm_lst = _res_df.apply(lambda row: row['Lambda'], axis=1).tolist()
        # remove the duplicate values from the list
        lm_lst = list(dict.fromkeys(lm_lst))
        
        for v in lm_lst:
            rslt_frf_df = _res_df[_res_df['Lambda'].isin([v])]
            x_val_1 = rslt_frf_df['Frequency'].to_list()
            y_val_1 = rslt_frf_df['Org_Mag'].to_list()
            y_val_2 = rslt_frf_df['Pred_Mag'].to_list()
            text1 = ['Freq : {}'.format(
                x_val_1[k]) + ',<br>Mag : {}'.format(y_val_1[k]) for k in range(len(x_val_1))]
            text2 = ['Freq : {}'.format(
                x_val_1[k]) + ',<br>Mag : {}'.format(y_val_2[k]) for k in range(len(x_val_1))]
            fig.add_trace(go.Scatter(
                x=x_val_1, y=y_val_1, name=f' TrueLambda= {v}',
                text=text1,
                hoverinfo='name+text',
                hoverlabel=dict(namelength=-1)))

            fig.add_trace(go.Scatter(
                x=x_val_1, y=y_val_2, name=f'PredLambda= {v}',
                text=text2,
                hoverinfo='name+text',
                hoverlabel=dict(namelength=-1)))
            
        # set the graph template in plotly
        large_rockwell_template = dict(
            layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
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
        
        # save plots
        fig.write_html(os.path.realpath(os.path.join(PathUtils.get_results_directory, 'true_pred_MagnitudePlot.html')))
        fig.write_image(os.path.realpath(os.path.join(PathUtils.get_results_directory, 'true_pred_MagnitudePlot.svg')))