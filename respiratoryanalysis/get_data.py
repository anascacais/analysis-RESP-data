# third-party
import pandas as pd
import pickle
import numpy as np

# local
from files import load_raw_data
from processing import preprocess

# built-in
import os
import json


def get_participant_ids(acquisition_folderpath):
    f = open(os.path.join(acquisition_folderpath, 'Cali_factors.json'))
    data = json.load(f)
    f.close()
    id_participants = data.keys()
    return id_participants


def get_data_by_id_activity(acquisition_folderpath, save=False):
    '''
    Returns
    -------
    data: dict
        Dictionary where keys are, recursively, participant ID and then activity. The last "value" corresponds to a pd.DataFrame with the resp signal timeseries for ['mag', 'airflow', 'pzt']

    '''

    id_participants = get_participant_ids(acquisition_folderpath)

    data = {}
    data_raw = {}
    print('Getting data for participants...')
    for id in id_participants:
        print(' ---------', id, '---------------')
        mag_data, airflow_data, pzt_data = load_raw_data(
            acquisition_folderpath, id)

        with open(os.path.join(acquisition_folderpath, id, f'idx_{id}.json'), "r") as jsonFile:
            activities_info = json.load(jsonFile)

        data[id] = {}
        data_raw[id] = {}

        for activity in activities_info.keys():

            data[id][activity] = pd.DataFrame(
                columns=['mag', 'airflow', 'pzt'])
            data_raw[id][activity] = pd.DataFrame(
                columns=['mag', 'airflow', 'pzt'])

            mag_data_4activity = mag_data['MAG'][activities_info[activity]['start_ind_scientisst']: activities_info[activity]['start_ind_scientisst'] + activities_info[activity]['length']]
            airflow_data_4activity = airflow_data['Airflow'][activities_info[activity]['start_ind_biopac']: activities_info[activity]['start_ind_biopac'] + activities_info[activity]['length']]
            pzt_data_4activity = pzt_data['PZT'][activities_info[activity]['start_ind_bitalino']: activities_info[activity]['start_ind_bitalino'] + activities_info[activity]['length']]

            mag_data_processed, airflow_data_processed, pzt_data_processed = preprocess(
                mag_data_4activity, airflow_data_4activity, pzt_data_4activity)

            data[id][activity]['mag'] = mag_data_processed
            data[id][activity]['airflow'] = airflow_data_processed
            data[id][activity]['pzt'] = pzt_data_processed

            data_raw[id][activity]['mag'] = mag_data_4activity.values
            data_raw[id][activity]['airflow'] = airflow_data_4activity.values
            data_raw[id][activity]['pzt'] = pzt_data_4activity.values

    if save:
        with open(os.path.join('Results', 'data_by_participant_activity.pickle'), 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    return data, data_raw


def get_performance_metrics(overview):
    '''
    Parameters
    ---------- 
    overview: pd.DataFrame
        Dataframe with the overview of the results for a specific participant activity and device

    Returns
    -------
    metrics: list
        List with the performance metrics [ratio, precision, recall]
    '''

    # inspiration
    ratio_i = (len(overview["TP_i"]) + len(overview["FP_i"])) / \
        (len(overview["TP_i"]) + len(overview["FN_i"]))
    precision_i = len(overview["TP_i"]) / \
        (len(overview["TP_i"]) + len(overview["FP_i"]))
    recall_i = len(overview["TP_i"]) / \
        (len(overview["TP_i"]) + len(overview["FN_i"]))

    # expiration
    ratio_e = (len(overview["TP_e"]) + len(overview["FP_e"])) / \
        (len(overview["TP_e"]) + len(overview["FN_e"]))
    precision_e = len(overview["TP_e"]) / \
        (len(overview["TP_e"]) + len(overview["FP_e"]))
    recall_e = len(overview["TP_e"]) / \
        (len(overview["TP_e"]) + len(overview["FN_e"]))

    # overall
    ratio = ((len(overview["TP_i"]) + len(overview["TP_e"])) + (len(overview["FP_i"]) + len(overview["FP_e"]))) / (
        (len(overview["TP_i"]) + len(overview["TP_e"])) + (len(overview["FN_i"]) + len(overview["FN_e"])))
    precision = (len(overview["TP_i"]) + len(overview["TP_e"])) / ((len(overview["TP_i"]) +
                                                                    len(overview["TP_e"])) + (len(overview["FP_i"]) + len(overview["FP_e"])))
    recall = (len(overview["TP_i"]) + len(overview["TP_e"])) / ((len(overview["TP_i"]) +
                                                                 len(overview["TP_e"])) + (len(overview["FN_i"]) + len(overview["FN_e"])))

    return [round(ratio, 2), round(precision, 2), round(recall, 2)]


def get_delays(overview):
    '''
    Parameters
    ---------- 
    overview: pd.DataFrame
        Dataframe with the overview of the results for a specific participant, activity and device
    overview_Airflow: pd.DataFrame
        Dataframe with the overview of the results for a specific participant and activity for the Airflow device

    Returns
    -------
    delays: list
        List with delay measures: ["mean absolute delay (std absolute delay)", delay adjusted to length of mean tB]. The absolute delay is in seconds
    '''

    # # inspiration
    # delays_i = pd.Series(overview["delay_i"])
    # adjusted_mean_delay_i = int(
    #     pd.Series(overview["adjusted_delay_i"]).mean() * 100)

    # # expiration
    # delay_e = pd.Series(overview["delay_e"])
    # adjusted_mean_delay_e = int(
    #     pd.Series(overview["adjusted_delay_e"]).mean() * 100)

    # overall
    delays = pd.Series(overview["delay_i"] + overview["delay_e"])
    adjusted_mean_delay = int(pd.Series(
        overview["adjusted_delay_i"] + overview["adjusted_delay_e"]).mean() * 100)

    return [f"{delays.mean():.2f} $\pm$ {delays.std().round(2):.2f}", f"{adjusted_mean_delay} \%"]


def get_respiratory_parameters(overview, overview_ref):
    '''
    Parameters
    ---------- 
    overview: pd.DataFrame
        Dataframe with the overview of the results for a specific participant, activity and device
    overview_ref: pd.DataFrame
        Dataframe with the overview of the airflow results for a specific participant and activity

    Returns
    -------
    resp_param: list
        List with respiratory parameters: ["inspiratory time", "expiratory time", "total breath time"]. All times are in seconds
    '''

    ti = pd.Series(overview["tI (s)"])
    te = pd.Series(overview["tE (s)"])
    tb = pd.Series(overview["tB (s)"])

    return [f"{ti.mean().round(2)} $\pm$ {ti.std().round(2)}", f"{te.mean().round(2)} $\pm$ {te.std().round(2)}", f"{tb.mean().round(2)} $\pm$ {tb.std().round(2)}"]


def transform_overview_on_target(overview, target, sampling_frequency=100):
    overview_transformed = {}

    for id in overview.keys():
        if target == "ID":
            key = id

        for activity in overview[id].keys():
            if target == "Activity":
                key = activity

            # get mean duration of respiratory cycle (tB) for each participant and activity (to compute adjusted delay)
            mean_tB = np.array(overview[id][activity]
                               ["Airflow"]["tB (s)"]).mean()

            if key not in overview_transformed.keys():
                overview_transformed[key] = {
                    "MAG": {}, "Airflow": {}, "PZT": {}}

            for metric in ["TP_i", "TP_e", "FP_i", "FP_e", "FN_i", "FN_e"]:
                overview_transformed[key]["MAG"][metric] = overview_transformed[key]["MAG"].get(
                    metric, []) + overview[id][activity]["MAG"][metric]
                overview_transformed[key]["PZT"][metric] = overview_transformed[key]["PZT"].get(
                    metric, []) + overview[id][activity]["PZT"][metric]

            for metric in ["tI (s)", "tE (s)", "tB (s)"]:
                overview_transformed[key]["MAG"][metric] = overview_transformed[key]["MAG"].get(
                    metric, []) + overview[id][activity]["MAG"][metric].tolist()
                overview_transformed[key]["PZT"][metric] = overview_transformed[key]["PZT"].get(
                    metric, []) + overview[id][activity]["PZT"][metric].tolist()

            for metric in ["tI airflow (s)", "tE airflow (s)", "tB airflow (s)"]:
                overview_transformed[key]["MAG"][metric] = overview_transformed[key]["MAG"].get(
                    metric, []) + overview[id][activity]["MAG"][metric].tolist()
                overview_transformed[key]["PZT"][metric] = overview_transformed[key]["PZT"].get(
                    metric, []) + overview[id][activity]["PZT"][metric].tolist()

            for metric in ["delay_i", "delay_e"]:
                delays_mag = (pd.Series(
                    overview[id][activity]["MAG"][metric]) * (-1/sampling_frequency)).tolist()
                delays_pzt = (pd.Series(
                    overview[id][activity]["PZT"][metric]) * (-1/sampling_frequency)).tolist()

                overview_transformed[key]["MAG"][metric] = overview_transformed[key]["MAG"].get(
                    metric, []) + delays_mag
                overview_transformed[key]["PZT"][metric] = overview_transformed[key]["PZT"].get(
                    metric, []) + delays_pzt
                overview_transformed[key]["MAG"][f"adjusted_{metric}"] = overview_transformed[key]["MAG"].get(
                    f"adjusted_{metric}", []) + [d / mean_tB for d in delays_mag]
                overview_transformed[key]["PZT"][f"adjusted_{metric}"] = overview_transformed[key]["PZT"].get(
                    f"adjusted_{metric}", []) + [d / mean_tB for d in delays_pzt]

    return overview_transformed


def transform_overview_on_overall(overview, sampling_frequency=100, ignore_key=None):

    overview_transformed = {"MAG": {}, "Airflow": {}, "PZT": {}}

    for id in overview.keys():

        if ignore_key is None or id != ignore_key:

            for activity in overview[id].keys():

                if ignore_key is None or activity != ignore_key:

                    # get mean duration of respiratory cycle (tB) for each participant and activity (to compute adjusted delay)
                    mean_tB = np.mean(
                        overview[id][activity]["Airflow"]["tB (s)"])

                    for metric in ["TP_i", "TP_e", "FP_i", "FP_e", "FN_i", "FN_e"]:
                        overview_transformed["MAG"][metric] = overview_transformed["MAG"].get(
                            metric, []) + overview[id][activity]["MAG"][metric]
                        overview_transformed["PZT"][metric] = overview_transformed["PZT"].get(
                            metric, []) + overview[id][activity]["PZT"][metric]

                    for metric in ["tI (s)", "tE (s)", "tB (s)"]:
                        overview_transformed["MAG"][metric] = overview_transformed["MAG"].get(
                            metric, []) + overview[id][activity]["MAG"][metric].tolist()
                        overview_transformed["PZT"][metric] = overview_transformed["PZT"].get(
                            metric, []) + overview[id][activity]["PZT"][metric].tolist()

                    for metric in ["tI airflow (s)", "tE airflow (s)", "tB airflow (s)"]:
                        overview_transformed["MAG"][metric] = overview_transformed["MAG"].get(
                            metric, []) + overview[id][activity]["MAG"][metric].tolist()
                        overview_transformed["PZT"][metric] = overview_transformed["PZT"].get(
                            metric, []) + overview[id][activity]["PZT"][metric].tolist()

                    for metric in ["delay_i", "delay_e"]:
                        delays_mag = (pd.Series(
                            overview[id][activity]["MAG"][metric]) * (-1/sampling_frequency)).tolist()
                        delays_pzt = (pd.Series(
                            overview[id][activity]["PZT"][metric]) * (-1/sampling_frequency)).tolist()

                        overview_transformed["MAG"][metric] = overview_transformed["MAG"].get(
                            metric, []) + delays_mag
                        overview_transformed["PZT"][metric] = overview_transformed["PZT"].get(
                            metric, []) + delays_pzt
                        overview_transformed["MAG"][f"adjusted_{metric}"] = overview_transformed["MAG"].get(
                            f"adjusted_{metric}", []) + [d / mean_tB for d in delays_mag]
                        overview_transformed["PZT"][f"adjusted_{metric}"] = overview_transformed["PZT"].get(
                            f"adjusted_{metric}", []) + [d / mean_tB for d in delays_pzt]

    return overview_transformed
