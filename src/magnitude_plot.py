import plotly.graph_objects as go
from utils.path_utils import PathUtils
import os

class MagnitudePlot:
    def __init__(self, df) -> None:
        self.df = df
        self._res_path = PathUtils().get_results_directory()
        self._plot()
        
    def _plot(self):
        plt_name = os.path.realpath(os.path.join(self._res_path, 'predicted_freq_plot.html'))
        large_rockwell_template = dict(
            layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
        # Create traces
        fig = go.Figure()

        # extract the 'Lambda'values from dataframe
        lm_lst = self.df.apply(lambda row: row['Lambda'], axis=1).tolist()

        # remove the duplicate values from the list
        lm_lst = list(dict.fromkeys(lm_lst))

        for b in lm_lst:
            if not self.df.empty:
                rslt_frf_df = self.df[self.df['Lambda'].isin([b])]
                x_val = rslt_frf_df['Frequency'].to_list()
                y_val = rslt_frf_df['Pred_Mag'].to_list()
                lm_name = f'Lambda = {b}'
                text = ['Freq : {}'.format(
                    x_val[k]) + ',<br>Mag : {}'.format(y_val[k]) for k in range(len(x_val))]
                fig.add_trace(go.Scatter(x=x_val, y=y_val,
                                         mode='lines',
                                         name=lm_name,
                                         text=text,
                                         hoverinfo='name+text',
                                         hoverlabel=dict(namelength=-1)))
                rslt_lst = rslt_frf_df.index
                self.df = self.df.drop(rslt_lst, axis=0)
            else:
                continue

        fig.update_layout(title='S-Parameters [Predicted Magnitude]',
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
        fig.write_image(os.path.realpath(os.path.join(self._res_path, 'predicted_freq_plot.svg')), scale=2)