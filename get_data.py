# third-party
import pandas as pd
import pickle 

# local
from files import load_data
from processing import preprocess

# built-in
import os
import json



def get_participant_ids():
    f = open(os.path.join('Aquisicao', 'Cali_factors.json'))
    data = json.load(f)
    f.close()
    id_participants = data.keys()
    return id_participants


def get_data_by_id_activity(save=False):
    '''
    Returns
    -------
    data: dict
        Dictionary where keys are, recursively, participant ID and then activity. The last "value" corresponds to a pd.DataFrame with the resp signal timeseries for ['scientisst', 'biopac', 'bitalino']
    
    ''' 

    id_participants = get_participant_ids()

    data = {}
    data_raw = {}

    for id in id_participants:
        print('---------',id,'---------------')
        scientisst_data, biopac_data, bitalino_data, activities_info = load_data('Aquisicao', id, resp_only=True)

        data[id] = {}
        data_raw[id] = {}

        for activity in activities_info.keys():

            data[id][activity] = pd.DataFrame(columns=['scientisst', 'biopac', 'bitalino'])
            data_raw[id][activity] = pd.DataFrame(columns=['scientisst', 'biopac', 'bitalino'])

            a = scientisst_data['RESP'][activities_info[activity]['start_ind_scientisst'] : activities_info[activity]['start_ind_scientisst'] + activities_info[activity]['length']]
            b = biopac_data['airflow'][activities_info[activity]['start_ind_biopac'] : activities_info[activity]['start_ind_biopac'] + activities_info[activity]['length']]
            c = bitalino_data['PZT'][activities_info[activity]['start_ind_bitalino'] : activities_info[activity]['start_ind_bitalino'] + activities_info[activity]['length']]

            scientisst_data_processed, _, bitalino_data_processed, biopac_data_processed = preprocess(a, b, c)

            data[id][activity]['scientisst'] = scientisst_data_processed
            data[id][activity]['biopac'] = biopac_data_processed
            data[id][activity]['bitalino'] = bitalino_data_processed

            data_raw[id][activity]['scientisst'] = a.values
            data_raw[id][activity]['biopac'] = b.values
            data_raw[id][activity]['bitalino'] = c.values


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
    ratio_i = (len(overview["TP_i"]) + len(overview["FP_i"])) / (len(overview["TP_i"]) + len(overview["FN_i"]))
    precision_i = len(overview["TP_i"]) / (len(overview["TP_i"]) + len(overview["FP_i"]))
    recall_i = len(overview["TP_i"]) / (len(overview["TP_i"]) + len(overview["FN_i"]))

    # expiration
    ratio_e = (len(overview["TP_e"]) + len(overview["FP_e"])) / (len(overview["TP_e"]) + len(overview["FN_e"]))
    precision_e = len(overview["TP_e"]) / (len(overview["TP_e"]) + len(overview["FP_e"]))
    recall_e = len(overview["TP_e"]) / (len(overview["TP_e"]) + len(overview["FN_e"]))
    
    # overall
    ratio = ((len(overview["TP_i"]) + len(overview["TP_e"])) + (len(overview["FP_i"]) + len(overview["FP_e"]))) / ((len(overview["TP_i"]) + len(overview["TP_e"])) + (len(overview["FN_i"]) + len(overview["FN_e"])))
    precision = (len(overview["TP_i"]) + len(overview["TP_e"])) / ((len(overview["TP_i"]) + len(overview["TP_e"])) + (len(overview["FP_i"]) + len(overview["FP_e"])))
    recall = (len(overview["TP_i"]) + len(overview["TP_e"])) / ((len(overview["TP_i"]) + len(overview["TP_e"])) + (len(overview["FN_i"]) + len(overview["FN_e"])))

    return [round(ratio, 2), round(precision, 2), round(recall, 2)]


def get_delays(overview):
    '''
    Parameters
    ---------- 
    overview: pd.DataFrame
        Dataframe with the overview of the results for a specific participant, activity and device
    overview_BIOPAC: pd.DataFrame
        Dataframe with the overview of the results for a specific participant and activity for the BIOPAC device
        
    Returns
    -------
    delays: list
        List with delay measures: ["mean absolute delay (std absolute delay)", delay adjusted to length of mean tB]. The absolute delay is in seconds
    ''' 

    # inspiration
    delays_i = pd.Series(overview["delay_i"])
    adjusted_mean_delay_i = int(pd.Series(overview["adjusted_delay_i"]).mean() * 100)

    # expiration
    delay_e = pd.Series(overview["delay_e"])
    adjusted_mean_delay_e = int(pd.Series(overview["adjusted_delay_e"]).mean() * 100)



    # overall
    delays = pd.Series(overview["delay_i"] + overview["delay_e"])
    adjusted_mean_delay = int(pd.Series(overview["adjusted_delay_i"] + overview["adjusted_delay_e"]).mean() * 100)

    return [f"{delays.mean().round(2)} $\pm$ {delays.std().round(2)}", f"{adjusted_mean_delay} %"]


def transform_overview_on_target(overview, target, sampling_frequency=100):
    overview_transformed = {}

    for id in overview.keys():
        if target == "ID":
            key = id

        for activity in overview[id].keys():
            if target == "Activity":
                key = activity

            # get mean duration of respiratory cycle (tB) for each participant and activity (to compute adjusted delay)
            mean_tB = overview[id][activity]["BIOPAC"]["tB (s)"].mean()

            if key not in overview_transformed.keys():
                overview_transformed[key] = {"ScientISST": {}, "BIOPAC": {}, "BITalino": {}}

            for metric in ["TP_i", "TP_e", "FP_i", "FP_e", "FN_i", "FN_e"]:
                overview_transformed[key]["ScientISST"][metric] = overview_transformed[key]["ScientISST"].get(metric, []) + overview[id][activity]["ScientISST"][metric]
                overview_transformed[key]["BITalino"][metric] = overview_transformed[key]["BITalino"].get(metric, []) + overview[id][activity]["BITalino"][metric]
            
            for metric in ["delay_i", "delay_e"]:
                delays_scientisst = (pd.Series(overview[id][activity]["ScientISST"][metric]) * (-1/sampling_frequency)).tolist()
                delays_bitalino = (pd.Series(overview[id][activity]["BITalino"][metric]) * (-1/sampling_frequency)).tolist()

                
                
                overview_transformed[key]["ScientISST"][metric] = overview_transformed[key]["ScientISST"].get(metric, []) + delays_scientisst
                overview_transformed[key]["BITalino"][metric] = overview_transformed[key]["BITalino"].get(metric, []) + delays_bitalino
                overview_transformed[key]["ScientISST"][f"adjusted_{metric}"] = overview_transformed[key]["ScientISST"].get(f"adjusted_{metric}", []) + [d / mean_tB for d in delays_scientisst]
                overview_transformed[key]["BITalino"][f"adjusted_{metric}"] = overview_transformed[key]["BITalino"].get(f"adjusted_{metric}", []) + [d / mean_tB for d in delays_bitalino]

    return overview_transformed


def transform_overview_on_overall(overview, sampling_frequency=100):

    overview_transformed = {"ScientISST": {}, "BIOPAC": {}, "BITalino": {}}

    for id in overview.keys():
        for activity in overview[id].keys():

            # get mean duration of respiratory cycle (tB) for each participant and activity (to compute adjusted delay)
            mean_tB = overview[id][activity]["BIOPAC"]["tB (s)"].mean()

            for metric in ["TP_i", "TP_e", "FP_i", "FP_e", "FN_i", "FN_e"]:
                overview_transformed["ScientISST"][metric] = overview_transformed["ScientISST"].get(metric, []) + overview[id][activity]["ScientISST"][metric]
                overview_transformed["BITalino"][metric] = overview_transformed["BITalino"].get(metric, []) + overview[id][activity]["BITalino"][metric]

            for metric in ["delay_i", "delay_e"]:
                delays_scientisst = (pd.Series(overview[id][activity]["ScientISST"][metric]) * (-1/sampling_frequency)).tolist()
                delays_bitalino = (pd.Series(overview[id][activity]["BITalino"][metric]) * (-1/sampling_frequency)).tolist()
                
                overview_transformed["ScientISST"][metric] = overview_transformed["ScientISST"].get(metric, []) + delays_scientisst
                overview_transformed["BITalino"][metric] = overview_transformed["BITalino"].get(metric, []) + delays_bitalino
                overview_transformed["ScientISST"][f"adjusted_{metric}"] = overview_transformed["ScientISST"].get(f"adjusted_{metric}", []) + [d / mean_tB for d in delays_scientisst]
                overview_transformed["BITalino"][f"adjusted_{metric}"] = overview_transformed["BITalino"].get(f"adjusted_{metric}", []) + [d / mean_tB for d in delays_bitalino]

    return overview_transformed