# build-in
import os
import json
import pickle

# third party
import pandas as pd

def load_raw_data(acquisition_folderpath, id):

    '''  brief description 
    
    Parameters
    ---------- 
    param1: int
         description
    
    Returns
    -------
    mag_data, airflow_data, pzt_data: array
        Full non-processed data for each device, in real units.
    
    ''' 

    airflow_data = pd.read_csv(os.path.join(acquisition_folderpath, id, f'biopac_{id}.txt'), sep='\t',  skiprows=11, index_col=False, names=["timestamp", "airflow", "ECG", "LED"])
    mag_data = pd.read_csv(os.path.join(acquisition_folderpath, id, f'scientisst_{id}.csv'), sep=',',  skiprows=2, index_col=False, names=["NSeq", "ECG", "ACC1", "ACC2", "ACC3", "LED", "RESP"], usecols=["ECG", "ACC1", "ACC2", "ACC3", "LED", "RESP"])
    pzt_data = pd.read_csv(os.path.join(acquisition_folderpath, id, f'bitalino_{id}.txt'), sep='\t',  skiprows=3, index_col=False, names=["nSeq", "I1", "I2", "O1", "O2", "PZT", "LUX", "A3", "A4", "A5", "A6"], usecols=["PZT", "LUX"])
    
    # to mv
    mag_data[["MAG"]] = ((mag_data[["RESP"]] * 3.3) / (2**12)) * 1000
    pzt_data[["PZT"]] = ((pzt_data[["PZT"]] * 3.3) / (2**10)) * 1000

    # to ml/s
    with open(os.path.join(acquisition_folderpath, "Cali_factors.json"), "r") as jsonFile:
        cali_factors = json.load(jsonFile)
    airflow_data[["Airflow"]] = airflow_data[["airflow"]] * cali_factors[id]

    airflow_data = airflow_data[["Airflow"]]
    mag_data = mag_data[["MAG"]]
    pzt_data = pzt_data[["PZT"]]

    
    # with open(os.path.join(directory, id, f'idx_{id}.json'), "r") as jsonFile:
    #     activities_info = json.load(jsonFile)
    
    return mag_data, airflow_data, pzt_data


def save_results(participant_results, id):
    
    try:
        results = pickle.load(open("Results/results.pickle", "rb"))
    except (OSError, IOError):
        results = {}
    
    finally:
        results[id] = participant_results

        with open('Results/results.pickle', 'wb') as file:
            pickle.dump(results, file)