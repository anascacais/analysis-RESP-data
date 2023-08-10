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

    id_participants = get_participant_ids()

    data = {}

    for id in id_participants:
        print('---------',id,'---------------')
        scientisst_data, biopac_data, bitalino_data, activities_info = load_data('Aquisicao', id, resp_only=True)

        data[id] = {}

        for activity in activities_info.keys():

            data[id][activity] = pd.DataFrame(columns=['scientisst', 'biopac', 'bitalino'])

            a = scientisst_data['RESP'][activities_info[activity]['start_ind_scientisst'] : activities_info[activity]['start_ind_scientisst'] + activities_info[activity]['length']]
            b = biopac_data['airflow'][activities_info[activity]['start_ind_biopac'] : activities_info[activity]['start_ind_biopac'] + activities_info[activity]['length']]
            c = bitalino_data['PZT'][activities_info[activity]['start_ind_bitalino'] : activities_info[activity]['start_ind_bitalino'] + activities_info[activity]['length']]

            scientisst_data_processed, _, bitalino_data_processed, biopac_data_processed = preprocess(a, b, c)

            data[id][activity]['scientisst'] = scientisst_data_processed
            data[id][activity]['biopac'] = biopac_data_processed
            data[id][activity]['bitalino'] = bitalino_data_processed


    if save:
        with open(os.path.join('Results', 'data_by_participant_activity.pickle'), 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL) 

    return data

