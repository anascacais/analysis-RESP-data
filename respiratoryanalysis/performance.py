# third-party
import pandas as pd
import numpy as np
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# local
from get_data import get_performance_metrics, get_delays
from respiratoryanalysis.constants import CATEGORICAL_PALETTE


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
            "Parameter", "Sensor", "Abs. error (s)", "Rel. error (%)", "Slope", "Intercept", "R^2"])

        for parameter in ["tI", "tE", "tB"]:
            print(
                f'N={len(overview["MAG"][f"{parameter} (s)"])} {parameter} for MAG')
            print(
                f'N={len(overview["PZT"][f"{parameter} (s)"])} {parameter} for PZT')

            for device in ["MAG", "PZT"]:
                new_entry = {}
                new_entry["Parameter"] = parameter
                new_entry["Sensor"] = device
                new_entry["Abs. error (s)"] = compute_mae(
                    overview[device][f"{parameter} (s)"], overview[device][f"{parameter} airflow (s)"])
                new_entry["Rel. error (%)"] = compute_mre(
                    overview[device][f"{parameter} (s)"], overview[device][f"{parameter} airflow (s)"])
                new_entry["R^2"], _, linreg = compute_r2_sse(
                    overview[device][f"{parameter} (s)"], overview[device][f"{parameter} airflow (s)"])
                new_entry["Slope"], new_entry["Intercept"] = f"{linreg.slope:.2f}", f"{linreg.intercept:.2f}"
                # new_entry["Bias (s)"], new_entry["Variability (s)"] = bland_altman_analysis(
                #     overview[device][f"{parameter} (s)"], overview[device][f"{parameter} airflow (s)"])

                breath_parameters.loc[len(breath_parameters)] = new_entry

    return breath_parameters


def compute_mae(test_param, target_param):
    if len(test_param) == 0:
        return "nan , nan"
    abs_error = np.abs(np.array(test_param) - np.array(target_param))
    return f"{np.mean(abs_error):.2f} $\pm$ {np.std(abs_error):.2f}"


def compute_mre(test_param, target_param):
    if len(test_param) == 0:
        return "nan , nan"
    abs_error = np.abs(np.array(test_param) - np.array(target_param))
    return f"{(np.mean(abs_error/np.array(target_param))*100):.2f} $\pm$ {(np.std(abs_error/np.array(target_param))*100):.2f}"


def compute_r2_sse(test_param, target_param):
    if len(test_param) == 0:
        return None, None, None

    try:
        linreg = stats.linregress(
            np.array(target_param), np.array(test_param))

    except Exception as e:
        print(e)

    slope = linreg.slope
    intercept = linreg.intercept
    r_value = linreg.rvalue

    y_pred = slope * np.array(target_param) + intercept
    sse = np.sum((np.array(test_param) - y_pred)**2)
    r_squared = r_value**2

    return np.around(r_squared, 2), np.around(sse, 2), linreg


def get_relative_errors(target_param, test_param):

    error = np.array(target_param) - np.array(test_param)
    return error / np.array(target_param)


def bland_altman_analysis(test_measures, target_measures):

    test_measures = np.array(test_measures)
    target_measures = np.array(target_measures)

    diff = test_measures - target_measures
    bias = np.mean(diff)
    variability = np.std(diff, ddof=1)

    return f"{bias:.3f}", f"{variability:.3f}"


def get_bias_variability(overview, target, metric, relative_error):

    if relative_error:
        unit = "%"
    else:
        unit = "s"

    if target in ["ID", "Activity"]:
        bias_variability = pd.DataFrame(columns=[
            target, "Sensor", f"Bias ({unit})", f"Variability ({unit})"])
        for key in overview.keys():
            for device in ["MAG", "PZT"]:
                new_entry = {}
                new_entry[target] = key
                new_entry["Sensor"] = device
                new_entry[f"Bias ({unit})"], new_entry[f"Variability ({unit})"] = bland_altman_analysis(
                    overview[key][device][f"{metric} (s)"], overview[key][device][f"{metric} airflow (s)"])

                bias_variability.loc[len(bias_variability)] = new_entry

    elif target == "both":
        bias_variability = pd.DataFrame(columns=[
            "ID", "Activity", "Sensor", f"Bias ({unit})", f"Variability ({unit})"])
        for id in overview.keys():
            for activity in overview[id].keys():
                for device in ["MAG", "PZT"]:
                    new_entry = {}
                    new_entry["ID"] = id
                    new_entry["Activity"] = activity
                    new_entry["Sensor"] = device
                    new_entry[f"Bias ({unit})"], new_entry[f"Variability ({unit})"] = bland_altman_analysis(
                        overview[id][activity][device][f"{metric} (s)"], overview[id][activity][device][f"{metric} airflow (s)"])

                    bias_variability.loc[len(bias_variability)] = new_entry
    else:
        bias_variability = pd.DataFrame(columns=[
            "Sensor", f"Bias ({unit})", f"Variability ({unit})"])
        for device in ["MAG", "PZT"]:
            new_entry = {}
            new_entry["Sensor"] = device
            new_entry[f"Bias ({unit})"], new_entry[f"Variability ({unit})"] = bland_altman_analysis(
                overview[device][f"{metric} (s)"], overview[device][f"{metric} airflow (s)"])

            bias_variability.loc[len(bias_variability)] = new_entry

    return bias_variability


def bland_altman_plot(mag_test_measures, mag_target_measures, pzt_test_measures, pzt_target_measures, metric, activity=None):

    test_measures = [np.array(mag_test_measures),
                     np.array(pzt_test_measures)]
    target_measures = [np.array(mag_target_measures),
                       np.array(pzt_target_measures)]

    fig = make_subplots(cols=2,
                        x_title=f"Mean of Airflow {metric} and sensor {metric} (s)",
                        y_title=f"Sensor {metric} - Airflow {metric} (s)",
                        subplot_titles=("MAG", "PZT"),
                        shared_yaxes=True)

    for i, sensor in enumerate(["MAG", "PZT"]):

        mean = np.mean([test_measures[i], target_measures[i]], axis=0)
        diff = (test_measures[i] - target_measures[i])
        md = np.mean(diff)
        sd = np.std(diff, ddof=1)

        fig.add_trace(go.Scatter(
            x=[min(mean), max(mean)],
            y=[md, md],
            mode="lines",
            line=dict(color=CATEGORICAL_PALETTE[1], width=2),
            name="Mean"
        ), row=1, col=i+1)

        # add upper and lower limits of agreement
        fig.add_trace(go.Scatter(
            x=[min(mean), max(mean)],
            y=[md + 1.96*sd, md + 1.96*sd],
            mode="lines",
            line=dict(color="black", dash="dash", width=2),
            name="LoA"
        ), row=1, col=i+1)
        fig.add_trace(go.Scatter(
            x=[min(mean), max(mean)],
            y=[md - 1.96*sd, md - 1.96*sd],
            mode="lines",
            line=dict(color="black", dash="dash", width=2)
        ), row=1, col=i+1)

        # Bland-Altman plot
        fig.add_trace(go.Scatter(
            x=mean,
            y=diff,
            mode="markers",
            marker_color=CATEGORICAL_PALETTE[0],
        ), row=1, col=i+1)

        print(
            f"{sensor} | mean: {md:.3f}, lloa: {md - 1.96*sd:.2f}, uloa: {md + 1.96*sd:.2f} ")

    for j, trace in enumerate(fig['data']):
        if j in [0, 1]:
            trace['showlegend'] = True
        else:
            trace['showlegend'] = False

    fig.update(layout_height=500, layout_width=600)
    if activity:
        fig.update_layout(
            title=f"Blant-Altman plot of {metric} (for {activity})",
            margin=go.layout.Margin(
                # b=50,  # bottom margin
                t=50,  # top margin
            )
        )
    else:
        fig.update_layout(yaxis_showticklabels=True,
                          yaxis2_showticklabels=False, title=f"Blant-Altman plot of {metric}",
                          margin=go.layout.Margin(
                              # b=10,  # bottom margin
                              t=50,  # top margin
                          )
                          )

    fig.show()
    if activity:
        fig.write_image(f"../results/blandaltman_{activity}_{metric}.png")
    else:
        fig.write_image(f"../results/blandaltman_{metric}.png")
