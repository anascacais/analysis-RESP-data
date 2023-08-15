# third-party
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import stats

# local
from color_palettes import CATEGORICAL_PALETTE

def normality_test(data, sensor, type):

    fig = make_subplots(cols=2, subplot_titles=(f'Histogram', f'Quantile-Quantile Plot'), specs=[[{"type": "bar"},  {"type": "scatter"}]])

    categorical_palette = iter(CATEGORICAL_PALETTE)
    fig.add_trace(go.Histogram(
        x=np.array(data), 
        marker_color=next(categorical_palette)
    ), row=1, col=1)

    qq = stats.probplot(data, dist='norm', sparams=(1))
    x = np.array([qq[0][0][0], qq[0][0][-1]])

    categorical_palette = iter(CATEGORICAL_PALETTE)

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
        title=f'{sensor} sensor ({type})', 
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
    

def plot_inspiration_vs_expiration(delays_df):
    ''' 
    Plot the distribution (as violin plots) of the delays for each sensor, comparing, side-by-side, the type of breath (Inhalation or Exhalation)

    Parameters
    ---------- 
    delays_df: pd.DataFrame
        Dataframe with individual delay instances, with 3 columns: [Sensor, Type, Delay], where "Sensor" is the sensor name, "Type" is the type of breath (Inhalation or Exhalation) and "Delay" is the delay in seconds.
    
    ''' 

    
    fig = go.Figure()
    categorical_palette = iter(CATEGORICAL_PALETTE)
    fig.add_trace(go.Violin(x=delays_df['Sensor'][delays_df['Type'] == 'Inhalation' ],
                            y=delays_df['Delay'][ delays_df['Type'] == 'Inhalation' ],
                            legendgroup='Inhalation', scalegroup='Inhalation', name='Inhalation',
                            side='negative',
                            line_color=next(categorical_palette))
                )
    fig.add_trace(go.Violin(x=delays_df['Sensor'][ delays_df['Type'] == 'Exhalation' ],
                            y=delays_df['Delay'][ delays_df['Type'] == 'Exhalation' ],
                            legendgroup='Exhalation', scalegroup='Exhalation', name='Exhalation',
                            side='positive',
                            line_color=next(categorical_palette))
                )
    fig.update_traces(meanline_visible=True)
    fig.update_layout(
        title=f'Delay distribution for each sensor, comparing Inhalation and Exhalation', 
        yaxis= {
            'title': 'Delay (in seconds)'
        },
        height=800, 
        width=900, 
        template='plotly_white', 
        violinmode='overlay'
    )
    #fig.update_layout(violingap=0, violinmode='overlay')
    fig.show()