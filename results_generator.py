# built-in
import json
import os

# third-party 
import numpy as np
from plotly import graph_objs as go

# local
from files import save_results
from processing import flow_reversal, time_compute, evaluate_extremums, compute_snr

def write_results(id, data_4id, data_raw_4id, acquisition_folderpath, show_fig=False):
    
    participant_results = {}

    with open(os.path.join(acquisition_folderpath, id, f'idx_{id}.json'), "r") as jsonFile:
        activities_info = json.load(jsonFile)
    
    for activity in activities_info.keys():

        if show_fig:
            fig = go.Figure()

        mag_data = data_4id[activity]['mag']
        airflow_data = data_4id[activity]['airflow']
        pzt_data = data_4id[activity]['pzt']

        airflow_data_raw = data_raw_4id[activity]['airflow']

        N = len(airflow_data_raw)
        time_x = np.linspace(0, N, N)

        # a = scientisst_data['RESP'][activities_info[activity]['start_ind_scientisst'] : activities_info[activity]['start_ind_scientisst'] + activities_info[activity]['length']]
        # b = biopac_data['airflow'][activities_info[activity]['start_ind_biopac'] : activities_info[activity]['start_ind_biopac'] + activities_info[activity]['length']]
        # c = bitalino_data['PZT'][activities_info[activity]['start_ind_bitalino'] : activities_info[activity]['start_ind_bitalino'] + activities_info[activity]['length']]

        # N = len(b)
        # time_x = np.linspace(0, N, N)

        # a, b, c, integral = preprocess(a, b, c)



        # plot
        if show_fig:
            fig.add_trace(go.Scatter(y = mag_data * 10, mode = 'lines', name='MAG'))
            fig.add_trace(go.Scatter(y = airflow_data, mode = 'lines', name='Airflow')) 
            fig.add_trace(go.Scatter(y = airflow_data_raw, mode = 'lines', name='Airflow (raw)')) 
            fig.add_trace(go.Scatter(y = pzt_data, mode = 'lines', name='PZT'))

        peaks_mag, valleys_mag = flow_reversal(mag_data)
        peaks_airflow, valleys_airflow = flow_reversal(airflow_data)
        peaks_pzt, valleys_pzt = flow_reversal(pzt_data)

        tb_airflow, ti_airflow , te_airflow, interval_airflow, ds_airflow = time_compute(peaks_airflow, valleys_airflow)
        br_airflow = (60 * len(tb_airflow)) / np.sum(tb_airflow)

        # evaluate peaks and valleys from MAG
        FP_s_e, TP_s_e, FN_s_e, performance_clf_s_e, positives_s_e, delay_s_e = evaluate_extremums(peaks_mag, peaks_airflow, tb_airflow, interval_airflow)
        FP_s_i, TP_s_i, FN_s_i, performance_clf_s_i, _, delay_s_i = evaluate_extremums(valleys_mag, valleys_airflow, tb_airflow, interval_airflow)
        # evaluate peaks and valleys from BITalino
        FP_c_e, TP_c_e, FN_c_e, _, _, delay_c_e = evaluate_extremums(peaks_pzt, peaks_airflow, tb_airflow, interval_airflow)
        FP_c_i, TP_c_i, FN_c_i, _, _, delay_c_i = evaluate_extremums(valleys_pzt, valleys_airflow, tb_airflow, interval_airflow)


        tb_mag, ti_mag , te_mag, _, ds_mag = time_compute(peaks_mag, valleys_mag)
        tb_pzt, ti_pzt , te_pzt, _, ds_pzt = time_compute(peaks_pzt, valleys_pzt)


        #print('Scientisst: tb', tb_a, 'ti' ,ti_a ,'te', te_a)
        br_mag = (60*len(tb_mag))/np.sum(tb_mag)
        br_pzt = (60*len(tb_pzt))/np.sum(tb_pzt)
        brv_mag = (np.std(tb_mag) / np.mean(tb_mag)) * 100
        brv_airflow = (np.std(tb_airflow) / np.mean(tb_airflow)) * 100
        brv_pzt = (np.std(tb_pzt) / np.mean(tb_pzt)) * 100

        if show_fig:
            for i in peaks_mag: 
                name = performance_clf_s_e[i]['clf']
                fig.add_vline(x=time_x[i], line_width=1, line_dash="dash", line_color="blue", name="", annotation_text=name)


            for i in valleys_mag: 
                name = performance_clf_s_i[i]['clf']
                fig.add_vline(x=time_x[i], line_width=1, line_dash="dash", line_color="green", name="", annotation_text=name)


            for j in peaks_airflow: 
                if j not in positives_s_e.keys():
                    name = 'FN'
                else: 
                    name = ''
                fig.add_vline(x=time_x[j], line_width=1, line_dash="dash", line_color="black", name="", annotation_text=name)

        
        #flow_reversal
        #ratio_bitalino = (len(ti_c)+len(te_c))/(len(ti_airflow)+len(te_airflow))
        #print((len(ti_mag)+len(te_mag))/(len(ti_airflow)+len(te_airflow)))
        if show_fig:
            fig.update_layout(title_text = activity)
            fig.show() 

        #save results:
        participant_results_activity = {
            'MAG':{
                'peaks': peaks_mag.tolist(),
                'valleys': valleys_mag.tolist(),
                'amplitude peaks': np.array(mag_data)[peaks_mag],
                'amplitude valleys': np.array(mag_data)[valleys_mag],
                'SNR': compute_snr(mag_data),
                'TP_i': TP_s_i,
                'FP_i': FP_s_i,
                'FN_i': FN_s_i,
                'TP_e': TP_s_e,
                'FP_e': FP_s_e,
                'FN_e': FN_s_e,
                'tI (s)': ti_mag,
                'tE (s)': te_mag,
                'tB (s)': tb_mag, 
                'delay_i': delay_s_i,
                'delay_e': delay_s_e,
                'ratio': (len(peaks_mag) + len(valleys_mag)) / (len(peaks_airflow) + len(valleys_airflow)),
                'ds (%)': ds_mag,
                'BR (bpm)': br_mag,
                'BRV (%)' : brv_mag,
            },
            'PZT':{
                'peaks': peaks_pzt.tolist(),
                'valleys': valleys_pzt.tolist(),
                'amplitude peaks':np.array(pzt_data)[peaks_pzt],
                'amplitude valleys':np.array(pzt_data)[valleys_pzt],
                'SNR': compute_snr(pzt_data),
                'TP_i': TP_c_i,
                'FP_i': FP_c_i,
                'FN_i': FN_c_i,
                'TP_e': TP_c_e,
                'FP_e': FP_c_e,
                'FN_e': FN_c_e,
                'delay_i': delay_c_i,
                'delay_e': delay_c_e,
                'tI (s)': ti_pzt,
                'tE (s)': te_pzt,
                'tB (s)': tb_pzt, 
                'ratio': (len(peaks_pzt) + len(valleys_pzt)) / (len(peaks_airflow) + len(valleys_airflow)),
                'BR (bpm)': br_pzt,
                'BRV (%)' : brv_pzt,
                'ds (%)': ds_pzt,
            },
            'Airflow':{
                'peaks': peaks_airflow.tolist(),
                'valleys': valleys_airflow.tolist(),
                'amplitude peaks':np.array(airflow_data_raw)[peaks_airflow],
                'amplitude valleys':np.array(airflow_data_raw)[valleys_airflow],
                'SNR': compute_snr(airflow_data_raw),
                'tI (s)': ti_airflow.tolist(),
                'tE (s)': te_airflow.tolist(),
                'tB (s)': tb_airflow, 
                'BRV (%)' : brv_airflow,
                'ds (%)': ds_airflow,
                'BR (bpm)': br_airflow,
            }
        }


        participant_results[activity] = participant_results_activity

    save_results(participant_results, id)