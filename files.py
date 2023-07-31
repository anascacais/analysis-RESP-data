# build-in
import os
import json
import pickle

# third party
import pandas as pd

def load_data(directory, id, resp_only=True):

    '''  brief description 
    
    Parameters
    ---------- 
    param1: int
         description
    
    Returns
    -------
    activities_info: dict 
        Dictionaire whose keys are the activities' names and values are the activity's duration in samples ("length"), 
        as well as start indexes for each device ("start_ind_bitalino", "start_ind_biopac", "start_ind_scientisst").
    
    ''' 

    biopac_data = pd.read_csv(os.path.join(directory, id, f'biopac_{id}.txt'), sep='\t',  skiprows=11, index_col=False, names=["timestamp", "airflow", "ECG", "LED"])
    scientisst_data = pd.read_csv(os.path.join(directory, id, f'scientisst_{id}.csv'), sep=',',  skiprows=2, index_col=False, names=["NSeq", "ECG", "ACC1", "ACC2", "ACC3", "LED", "RESP"], usecols=["ECG", "ACC1", "ACC2", "ACC3", "LED", "RESP"])
    bitalino_data = pd.read_csv(os.path.join(directory, id, f'bitalino_{id}.txt'), sep='\t',  skiprows=3, index_col=False, names=["nSeq", "I1", "I2", "O1", "O2", "PZT", "LUX", "A3", "A4", "A5", "A6"], usecols=["PZT", "LUX"])
    
    scientisst_data.RESP = -scientisst_data.RESP
    
    if resp_only:
        biopac_data = biopac_data[["airflow"]]
        scientisst_data = scientisst_data[["RESP"]]
        bitalino_data = bitalino_data[["PZT"]]

    
    with open(os.path.join(directory, id, f'idx_{id}.json'), "r") as jsonFile:
        activities_info = json.load(jsonFile)
    
    return scientisst_data, biopac_data, bitalino_data, activities_info


def save_results(participant_results, id):
    
    try:
        results = pickle.load(open("Results/results.pickle", "rb"))
    except (OSError, IOError):
        results = {}
    
    finally:
        results[id] = participant_results

        with open('Results/results.pickle', 'wb') as file:
            pickle.dump(results, file)