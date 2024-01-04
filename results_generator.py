# built-in
import json
import os

# third-party
import numpy as np
from plotly import graph_objs as go

# local
from files import save_results
from processing import flow_reversal, time_compute, evaluate_extremums, compute_snr, time_compute_prev


def minmax(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (0.5 * (max_val - min_val)) - 1
    return normalized_data


def write_results(id, data_4id, data_raw_4id, acquisition_folderpath, show_fig=True):

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

        peaks_mag, valleys_mag = flow_reversal(mag_data)
        peaks_airflow, valleys_airflow = flow_reversal(airflow_data)
        peaks_pzt, valleys_pzt = flow_reversal(pzt_data)

        tb_airflow, ti_airflow, te_airflow, interval_airflow, ds_airflow = time_compute_prev(
            peaks_airflow, valleys_airflow)
        br_airflow = (60 * len(tb_airflow)) / np.sum(tb_airflow)

        if activity == "ALR" and id == "QMQ7":
            print("ALR")

        # evaluate peaks and valleys from MAG
        FP_s_e, TP_s_e, FN_s_e, performance_clf_s_e, positives_s_e, delay_s_e = evaluate_extremums(
            peaks_mag, peaks_airflow, tb_airflow, interval_airflow)
        FP_s_i, TP_s_i, FN_s_i, performance_clf_s_i, positives_s_i, delay_s_i = evaluate_extremums(
            valleys_mag, valleys_airflow, tb_airflow, interval_airflow)
        # evaluate peaks and valleys from BITalino
        FP_c_e, TP_c_e, FN_c_e, _, positives_c_e, delay_c_e = evaluate_extremums(
            peaks_pzt, peaks_airflow, tb_airflow, interval_airflow)
        FP_c_i, TP_c_i, FN_c_i, _, positives_c_i, delay_c_i = evaluate_extremums(
            valleys_pzt, valleys_airflow, tb_airflow, interval_airflow)

        if show_fig:
            mag_norm = minmax(mag_data)
            airflow_norm = minmax(airflow_data)
            fig.add_trace(go.Scatter(y=mag_norm,
                          mode='lines', name='MAG'))
            fig.add_trace(go.Scatter(y=airflow_norm,
                          mode='lines', name='Airflow'))
            fig.add_trace(go.Scatter(
                x=TP_s_e, y=mag_norm[TP_s_e], mode='markers', name='TP start exp'))
            fig.add_trace(go.Scatter(
                x=FN_s_e, y=mag_norm[FN_s_e], mode='markers', name='FN start exp'))
            fig.add_trace(go.Scatter(
                x=TP_s_i, y=mag_norm[TP_s_i], mode='markers', name='TP start ins'))
            fig.add_trace(go.Scatter(
                x=FN_s_i, y=mag_norm[FN_s_i], mode='markers', name='FN start ins'))
            fig.show()

        tb_mag, ti_mag, te_mag, tb_mag_airflow, ti_mag_airflow, te_mag_airflow = time_compute(
            np.array(TP_s_e), np.array(TP_s_i), np.array(FN_s_e), np.array(FN_s_i), positives_s_e, positives_s_i)
        tb_pzt, ti_pzt, te_pzt, tb_pzt_airflow, ti_pzt_airflow, te_pzt_airflow = time_compute(
            np.array(TP_c_e), np.array(TP_c_i), np.array(FN_c_e), np.array(FN_c_i), positives_c_e, positives_c_i)

        # print('Scientisst: tb', tb_a, 'ti' ,ti_a ,'te', te_a)
        if len(tb_mag) != 0:
            br_mag = (60*len(tb_mag))/np.sum(tb_mag)
            brv_mag = (np.std(tb_mag) / np.mean(tb_mag)) * 100
        else:
            br_mag, brv_mag = None, None
        if len(tb_pzt) != 0:
            br_pzt = (60*len(tb_pzt))/np.sum(tb_pzt)
            brv_pzt = (np.std(tb_pzt) / np.mean(tb_pzt)) * 100
        else:
            br_pzt, brv_pzt = None, None

        brv_airflow = (np.std(tb_airflow) / np.mean(tb_airflow)) * 100

        # save results:
        participant_results_activity = {
            'MAG': {
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
                'tI airflow (s)': ti_mag_airflow,
                'tE airflow (s)': te_mag_airflow,
                'tB airflow (s)': tb_mag_airflow,
                'delay_i': delay_s_i,
                'delay_e': delay_s_e,
                'ratio': (len(peaks_mag) + len(valleys_mag)) / (len(peaks_airflow) + len(valleys_airflow)),
                # 'ds (%)': ds_mag,
                # 'ds airflow (%)': ds_mag_airflow,
                'BR (bpm)': br_mag,
                'BRV (%)': brv_mag
            },
            'PZT': {
                'peaks': peaks_pzt.tolist(),
                'valleys': valleys_pzt.tolist(),
                'amplitude peaks': np.array(pzt_data)[peaks_pzt],
                'amplitude valleys': np.array(pzt_data)[valleys_pzt],
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
                'tI airflow (s)': ti_pzt_airflow,
                'tE airflow (s)': te_pzt_airflow,
                'tB airflow (s)': tb_pzt_airflow,
                'ratio': (len(peaks_pzt) + len(valleys_pzt)) / (len(peaks_airflow) + len(valleys_airflow)),
                'BR (bpm)': br_pzt,
                'BRV (%)': brv_pzt,
                # 'ds (%)': ds_pzt,
                # 'ds airflow (%)': ds_pzt_airflow,
                'TP_i_indx': positives_c_i,
                'TP_e_indx': positives_c_e,
            },
            'Airflow': {
                'peaks': peaks_airflow.tolist(),
                'valleys': valleys_airflow.tolist(),
                'amplitude peaks': np.array(airflow_data_raw)[peaks_airflow],
                'amplitude valleys': np.array(airflow_data_raw)[valleys_airflow],
                'SNR': compute_snr(airflow_data_raw),
                'tI (s)': ti_airflow.tolist(),
                'tE (s)': te_airflow.tolist(),
                'tB (s)': tb_airflow.tolist(),
                'BRV (%)': brv_airflow,
                'ds (%)': ds_airflow,
                'BR (bpm)': br_airflow,
            }
        }

        participant_results[activity] = participant_results_activity

    save_results(participant_results, id)
