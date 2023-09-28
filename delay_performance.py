# third-party
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import stats

import plotly.io as pio   
pio.kaleido.scope.mathjax = None

# local
from constants import CATEGORICAL_PALETTE

def normality_test(data, sensor, type, categorical_palette=None):

    fig = make_subplots(cols=2, subplot_titles=(f'Histogram', f'Quantile-Quantile Plot'), specs=[[{"type": "bar"},  {"type": "scatter"}]])

    if not categorical_palette:
        categorical_palette = iter(CATEGORICAL_PALETTE)
    else:
        categorical_palette = iter(categorical_palette)

    fig.add_trace(go.Histogram(
        x=np.array(data), 
        marker_color=next(categorical_palette)
    ), row=1, col=1)

    qq = stats.probplot(data, dist='norm', sparams=(1))
    x = np.array([qq[0][0][0], qq[0][0][-1]])

    if not categorical_palette:
        categorical_palette = iter(CATEGORICAL_PALETTE)
    else:
        categorical_palette = iter(categorical_palette)

    fig.add_trace({
        'type': 'scatter',
        'x': qq[0][0], 
        'y': qq[0][1],
        'mode': 'markers',
        'marker': {
            'color': next(categorical_palette)
        }
    }, row=1, col=2)

    fig.add_trace({
        'type': 'scatter',
        'x': x, 
        'y': qq[1][1] + qq[1][0] * x,
        'mode': 'lines',
        'line': {
            'color': next(categorical_palette)
        }

    }, row=1, col=2)

    fig.update_layout(
        title=f'{sensor} sensor ({type}, N={len(data)})',
        xaxis= {
            'title': 'Delay',
            'zeroline': False
        },
        yaxis= {
            'title': 'Count'
        },
        xaxis2= {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        yaxis2= {
            'title': 'Sample Quantities'
        },
        height=800, 
        width=900, 
        showlegend=False, 
        template='plotly_white')

    fig.show()
    

def plot_inspiration_vs_expiration(delays_df, categorical_palette=None):
    ''' 
    Plot the distribution (as violin plots) of the delays for each sensor, comparing, side-by-side, the type of breath (Inhalation or Exhalation)

    Parameters
    ---------- 
    delays_df: pd.DataFrame
        Dataframe with individual delay instances, with 3 columns: [Sensor, Type, Delay], where "Sensor" is the sensor name, "Type" is the type of breath (Inhalation or Exhalation) and "Delay" is the delay in seconds.
    
    ''' 

    
    fig = go.Figure()
    
    if not categorical_palette:
        categorical_palette = iter(CATEGORICAL_PALETTE)
    else:
        categorical_palette = iter(categorical_palette)

    fig.add_trace(go.Violin(x=delays_df['Sensor'][delays_df['Type'] == 'Inhalation' ],
                            y=delays_df['Delay'][ delays_df['Type'] == 'Inhalation' ],
                            legendgroup='Inspiration', scalegroup='Inhalation', name='Inspiration',
                            side='negative',
                            line_color=next(categorical_palette))
                )
    fig.add_trace(go.Violin(x=delays_df['Sensor'][ delays_df['Type'] == 'Exhalation' ],
                            y=delays_df['Delay'][ delays_df['Type'] == 'Exhalation' ],
                            legendgroup='Expiration', scalegroup='Exhalation', name='Expiration',
                            side='positive',
                            line_color=next(categorical_palette))
                )
    fig.update_traces(meanline_visible=True, width=1.2)
    fig.update_layout(
        title=f'Delay distribution for each sensor (inspiration vs expiration)', 
        xaxis = {
            'title': 'Sensors'
        },
        yaxis = {
            'title': 'Delay (in seconds)'
        },
        height=600, 
        width=600, 
        template='plotly_white', 
        violinmode='overlay'
    )
    #fig.update_layout(violingap=0, violinmode='overlay')
    fig.show()
    fig.write_image("Results/delay_violin_dist.pdf")