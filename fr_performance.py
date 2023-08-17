# third-party
import pandas as pd

# local
from get_data import get_performance_metrics, get_delays

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
        
    sensor = {"ScientISST": "MAG", "BITalino": "PZT"}
    
    if target in ["ID", "Activity"]:
        fr_detection = pd.DataFrame(columns=[target, "Sensor", "Ratio", "Precision", "Recall", "Mean absolute delay $\pm$ SD", "Adjusted delay"])
        for key in overview.keys():
            for device in ["ScientISST", "BITalino"]:
                new_entry = {}
                new_entry[target] = key
                new_entry["Sensor"] = sensor[device]
                new_entry["Ratio"], new_entry["Precision"], new_entry["Recall"]  = get_performance_metrics(overview[key][device])
                new_entry["Mean absolute delay $\pm$ SD"], new_entry["Adjusted delay"] = get_delays(overview[key][device])
                fr_detection.loc[len(fr_detection)] = new_entry

    elif target == "both":
        fr_detection = pd.DataFrame(columns=["ID", "Activity", "Sensor", "Ratio", "Precision", "Recall"])
        for id in overview.keys():
            for activity in overview[id].keys():
                for device in ["ScientISST", "BITalino"]:
                    new_entry = {}
                    new_entry["ID"] = id
                    new_entry["Activity"] = activity
                    new_entry["Sensor"] = sensor[device]
                    new_entry["Ratio"], new_entry["Precision"], new_entry["Recall"]  = get_performance_metrics(overview[id][activity][device])
                    #new_entry["Mean absolute delay $\pm$ SD"], new_entry["Adjusted delay"] = get_delays(overview[id][activity][device])
                    fr_detection.loc[len(fr_detection)] = new_entry


    else:
        fr_detection = pd.DataFrame(columns=["Sensor", "Ratio", "Precision", "Recall", "Mean absolute delay $\pm$ SD", "Adjusted delay"])
        for device in ["ScientISST", "BITalino"]:
            new_entry = {}
            new_entry["Sensor"] = sensor[device]
            new_entry["Ratio"], new_entry["Precision"], new_entry["Recall"]  = get_performance_metrics(overview[device])
            new_entry["Mean absolute delay $\pm$ SD"], new_entry["Adjusted delay"] = get_delays(overview[device])
            fr_detection.loc[len(fr_detection)] = new_entry


    return fr_detection
