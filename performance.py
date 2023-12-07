# third-party
import pandas as pd
import numpy as np
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# local
from get_data import get_performance_metrics, get_delays
from constants import CATEGORICAL_PALETTE


def get_fr_detection_performance(overview, target):
    '''
    Parameters
    ---------- 
    overview: pd.DataFrame
        Dataframe with the overview of the results for a specific device and target
    target: str
        Target of the analysis. Can be "ID", "Activity", "both" or None. 

    Returns
    -------
    fr_detection: pd.DataFrame
        Dataframe with the performance metrics (Ratio, Precision, Recall, Mean absolute delay +/- SD and Adjusted delay), accroding to sensor and target

    '''

    if target in ["ID", "Activity"]:
        fr_detection = pd.DataFrame(columns=[
                                    target, "Sensor", "Ratio", "Precision", "Recall", "Mean absolute delay $\pm$ SD", "Adjusted delay"])
        for key in overview.keys():
            for device in ["MAG", "PZT"]:
                new_entry = {}
                new_entry[target] = key
                new_entry["Sensor"] = device
                new_entry["Ratio"], new_entry["Precision"], new_entry["Recall"] = get_performance_metrics(
                    overview[key][device])
                new_entry["Mean absolute delay $\pm$ SD"], new_entry["Adjusted delay"] = get_delays(
                    overview[key][device])
                fr_detection.loc[len(fr_detection)] = new_entry

    elif target == "both":
        fr_detection = pd.DataFrame(
            columns=["ID", "Activity", "Sensor", "Ratio", "Precision", "Recall"])
        for id in overview.keys():
            for activity in overview[id].keys():
                for device in ["MAG", "PZT"]:
                    new_entry = {}
                    new_entry["ID"] = id
                    new_entry["Activity"] = activity
                    new_entry["Sensor"] = device
                    new_entry["Ratio"], new_entry["Precision"], new_entry["Recall"] = get_performance_metrics(
                        overview[id][activity][device])
                    # new_entry["Mean absolute delay $\pm$ SD"], new_entry["Adjusted delay"] = get_delays(overview[id][activity][device])
                    fr_detection.loc[len(fr_detection)] = new_entry

    else:
        fr_detection = pd.DataFrame(columns=[
                                    "Sensor", "Ratio", "Precision", "Recall", "Mean absolute delay $\pm$ SD", "Adjusted delay"])
        for device in ["MAG", "PZT"]:
            new_entry = {}
            new_entry["Sensor"] = device
            new_entry["Ratio"], new_entry["Precision"], new_entry["Recall"] = get_performance_metrics(
                overview[device])
            # new_entry["Mean absolute delay $\pm$ SD"], new_entry["Adjusted delay"] = get_delays(overview[device])
            new_entry["Mean absolute delay $\pm$ SD"], _ = get_delays(
                overview[device])  # removed adjusted delay
            fr_detection.loc[len(fr_detection)] = new_entry

    return fr_detection


def get_breath_parameters_performance(overview, target):

    if target in ["ID", "Activity"]:
        breath_parameters = pd.DataFrame(columns=[
            target, "Sensor", "MAE Ti (s)", "MRE Ti (%)", "MAE Te (s)", "MRE Te (%)", "MAE Tb (s)", "MRE Tb (%)"])
        for key in overview.keys():
            for device in ["MAG", "PZT"]:
                new_entry = {}
                new_entry[target] = key
                new_entry["Sensor"] = device
                new_entry["MAE Ti (s)"], new_entry["MAE Te (s)"], new_entry["MAE Tb (s)"] = compute_mae(overview[key][device]["tI (s)"], overview[key][device]["tI airflow (s)"]), compute_mae(
                    overview[key][device]["tE (s)"], overview[key][device]["tE airflow (s)"]), compute_mae(overview[key][device]["tB (s)"], overview[key][device]["tB airflow (s)"])
                new_entry["MRE Ti (%)"], new_entry["MRE Te (%)"], new_entry["MRE Tb (%)"] = compute_mre(overview[key][device]["tI (s)"], overview[key][device]["tI airflow (s)"]), compute_mre(
                    overview[key][device]["tE (s)"], overview[key][device]["tE airflow (s)"]), compute_mre(overview[key][device]["tB (s)"], overview[key][device]["tB airflow (s)"])
                # new_entry["R^2 Ti"], new_entry["SSE Ti"], _ = compute_r2_sse(
                #     overview[key][device]["tI (s)"], overview[key][device]["tI airflow (s)"])
                # new_entry["R^2 Te"], new_entry["SSE Te"], _ = compute_r2_sse(
                #     overview[key][device]["tE (s)"], overview[key][device]["tE airflow (s)"])
                # new_entry["R^2 Tb"], new_entry["SSE Tb"], _ = compute_r2_sse(
                #     overview[key][device]["tB (s)"], overview[key][device]["tB airflow (s)"])
                breath_parameters.loc[len(breath_parameters)] = new_entry

    elif target == "both":
        breath_parameters = pd.DataFrame(columns=[
            "ID", "Activity", "Sensor", "MAE Ti (s)", "MRE Ti (%)", "MAE Te (s)", "MRE Te (%)", "MAE Tb (s)", "MRE Tb (%)"])
        for id in overview.keys():
            for activity in overview[id].keys():
                for device in ["MAG", "PZT"]:
                    new_entry = {}
                    new_entry["ID"] = id
                    new_entry["Activity"] = activity
                    new_entry["Sensor"] = device
                    new_entry["MAE Ti (s)"], new_entry["MAE Te (s)"], new_entry["MAE Tb (s)"] = compute_mae(overview[id][activity][device]["tI (s)"], overview[id][activity][device]["tI airflow (s)"]), compute_mae(
                        overview[id][activity][device]["tE (s)"], overview[id][activity][device]["tE airflow (s)"]), compute_mae(overview[id][activity][device]["tB (s)"], overview[id][activity][device]["tB airflow (s)"])
                    new_entry["MRE Ti (%)"], new_entry["MRE Te (%)"], new_entry["MRE Tb (%)"] = compute_mre(overview[id][activity][device]["tI (s)"], overview[id][activity][device]["tI airflow (s)"]), compute_mre(
                        overview[id][activity][device]["tE (s)"], overview[id][activity][device]["tE airflow (s)"]), compute_mre(overview[id][activity][device]["tB (s)"], overview[id][activity][device]["tB airflow (s)"])
                    # new_entry["R^2 Ti"], new_entry["SSE Ti"], _ = compute_r2_sse(
                    #     overview[id][activity][device]["tI (s)"], overview[id][activity][device]["tI airflow (s)"])
                    # new_entry["R^2 Te"], new_entry["SSE Te"], _ = compute_r2_sse(
                    #     overview[id][activity][device]["tE (s)"], overview[id][activity][device]["tE airflow (s)"])
                    # new_entry["R^2 Tb"], new_entry["SSE Tb"], _ = compute_r2_sse(
                    #     overview[id][activity][device]["tB (s)"], overview[id][activity][device]["tB airflow (s)"])
                    breath_parameters.loc[len(breath_parameters)] = new_entry

    else:
        breath_parameters = pd.DataFrame(columns=[
            "Sensor", "MAE Ti (s)", "MRE Ti (%)", "MAE Te (s)", "MRE Te (%)", "MAE Tb (s)", "MRE Tb (%)"])

        for device in ["MAG", "PZT"]:
            new_entry = {}
            new_entry["Sensor"] = device
            new_entry["MAE Ti (s)"], new_entry["MAE Te (s)"], new_entry["MAE Tb (s)"] = compute_mae(overview[device]["tI (s)"], overview[device]["tI airflow (s)"]), compute_mae(
                overview[device]["tE (s)"], overview[device]["tE airflow (s)"]), compute_mae(overview[device]["tB (s)"], overview[device]["tB airflow (s)"])
            new_entry["MRE Ti (%)"], new_entry["MRE Te (%)"], new_entry["MRE Tb (%)"] = compute_mre(overview[device]["tI (s)"], overview[device]["tI airflow (s)"]), compute_mre(
                overview[device]["tE (s)"], overview[device]["tE airflow (s)"]), compute_mre(overview[device]["tB (s)"], overview[device]["tB airflow (s)"])
            # new_entry["R^2 Ti"], new_entry["SSE Ti"], _ = compute_r2_sse(
            #     overview[device]["tI (s)"], overview[device]["tI airflow (s)"])
            # new_entry["R^2 Te"], new_entry["SSE Te"], _ = compute_r2_sse(
            #     overview[device]["tE (s)"], overview[device]["tE airflow (s)"])
            # new_entry["R^2 Tb"], new_entry["SSE Tb"], _ = compute_r2_sse(
            #     overview[device]["tB (s)"], overview[device]["tB airflow (s)"])
            breath_parameters.loc[len(breath_parameters)] = new_entry

    return breath_parameters


def compute_mae(test_param, target_param):

    abs_error = np.abs(np.array(test_param) - np.array(target_param))
    return f"{np.around(np.mean(abs_error), 2)} , {np.around(np.std(abs_error), 2)}"


def compute_mre(test_param, target_param):

    abs_error = np.abs(np.array(test_param) - np.array(target_param))
    return f"{np.around(np.mean(abs_error/np.array(target_param))*100, 2)} , {np.around(np.std(abs_error/np.array(target_param))*100, 2)}"


def compute_r2_sse(test_param, target_param):

    linreg = stats.linregress(
        np.array(target_param), np.array(test_param))

    slope = linreg.slope
    intercept = linreg.intercept
    r_value = linreg.rvalue

    y_pred = slope * np.array(target_param) + intercept
    sse = np.sum((np.array(test_param) - y_pred)**2)
    r_squared = r_value**2

    return np.around(r_squared, 2), np.around(sse, 2), linreg


def bland_altman_plot(test_measures, target_measures, metric=None, sensor=None):

    test_measures = np.array(test_measures)
    target_measures = np.array(target_measures)

    mean = np.mean([test_measures, target_measures], axis=0)
    diff = test_measures - target_measures
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    fig = make_subplots(cols=2, subplot_titles=(
        f"Linear regression of {sensor} {metric} by Airflow {metric}", f"Bland-Altman plot of {metric}"))

    r_squared, _, linreg = compute_r2_sse(test_measures, target_measures)
    fig.add_trace(go.Scatter(
        x=target_measures,
        y=linreg.intercept + linreg.slope*target_measures,
        mode="lines",
        line=dict(color=CATEGORICAL_PALETTE[1], width=1),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=target_measures,
        y=target_measures,
        mode="lines",
        line=dict(color="black", dash="dash", width=1),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=target_measures,
        y=test_measures,
        mode="markers",
        marker_color=CATEGORICAL_PALETTE[0]
    ), row=1, col=1)

    fig.add_annotation(
        text=f"R^2 = {r_squared}",
        showarrow=False,
        yanchor='bottom',
        xanchor='right',
        xshift=350,
        yshift=500,
        font=dict(
            # size=16,
            color=CATEGORICAL_PALETTE[1]
        ),
        row=1, col=1)

    # add mean line
    fig.add_trace(go.Scatter(
        x=[min(mean), max(mean)],
        y=[md, md],
        mode="lines",
        line=dict(color=CATEGORICAL_PALETTE[1], width=1)
    ), row=1, col=2)

    # add upper and lower limits of agreement
    fig.add_trace(go.Scatter(
        x=[min(mean), max(mean)],
        y=[md + 1.96*sd, md + 1.96*sd],
        mode="lines",
        line=dict(color="black", dash="dash", width=1)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[min(mean), max(mean)],
        y=[md - 1.96*sd, md - 1.96*sd],
        mode="lines",
        line=dict(color="black", dash="dash", width=1)
    ), row=1, col=2)

    # Bland-Altman plot
    fig.add_trace(go.Scatter(
        x=mean,
        y=diff,
        mode="markers",
        marker_color=CATEGORICAL_PALETTE[0]
    ), row=1, col=2)

    fig.update_xaxes(title_text=f"Airflow {metric} (s)", row=1, col=1)
    fig.update_yaxes(title_text=f"{sensor} {metric} (s)", row=1, col=1)
    fig.update_xaxes(
        title_text=f"Mean of Airflow {metric} and {sensor} {metric} (s)", row=1, col=2)
    fig.update_yaxes(
        title_text=f"{sensor} {metric} - Airflow {metric} (s)", row=1, col=2)

    fig.update(layout_showlegend=False, layout_height=800, layout_width=1200)
    fig.show()
