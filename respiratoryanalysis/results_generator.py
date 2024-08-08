# built-in
import json
import os

# third-party
import numpy as np
from plotly import graph_objs as go

# local
from files import save_results
from processing import flow_reversal, time_compute, evaluate_extrema, compute_snr, time_compute_prev


def minmax(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (0.5 * (max_val - min_val)) - 1
    return normalized_data


def write_results(id, data_4id, data_raw_4id, acquisition_folderpath, show_fig=True, threshold_acceptability=0.5):

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

        if activity == "SGB" and id == "EPE2":
            print("ALR")

        FP_mag, TP_mag, FN_mag, positives_mag, delays_mag = evaluate_extrema(
            peaks_mag, peaks_airflow, valleys_mag, valleys_airflow, tb_airflow, interval_airflow, threshold_acceptability
        )

        FP_pzt, TP_pzt, FN_pzt, positives_pzt, delays_pzt = evaluate_extrema(
            peaks_pzt, peaks_airflow, valleys_pzt, valleys_airflow, tb_airflow, interval_airflow, threshold_acceptability
        )

        if show_fig:
            mag_norm = minmax(mag_data)
            airflow_norm = minmax(airflow_data)
            fig.add_trace(go.Scatter(y=mag_norm,
                          mode='lines', name='MAG'))
            fig.add_trace(go.Scatter(y=airflow_norm,
                          mode='lines', name='Airflow'))
            fig.add_trace(go.Scatter(
                x=TP_mag["exp"], y=mag_norm[TP_mag["exp"]], mode='markers', name='TP start exp'))
            fig.add_trace(go.Scatter(
                x=FN_mag["exp"], y=mag_norm[FN_mag["exp"]], mode='markers', name='FN start exp'))
            fig.add_trace(go.Scatter(
                x=TP_mag["insp"], y=mag_norm[TP_mag["insp"]], mode='markers', name='TP start ins'))
            fig.add_trace(go.Scatter(
                x=FN_mag["insp"], y=mag_norm[FN_mag["insp"]], mode='markers', name='FN start ins'))
            fig.show()

        tb_mag, ti_mag, te_mag, tb_mag_airflow, ti_mag_airflow, te_mag_airflow = time_compute(
            np.array(TP_mag["exp"]), np.array(TP_mag["insp"]), np.array(FN_mag["exp"]), np.array(FN_mag["insp"]), positives_mag["exp"], positives_mag["insp"])
        tb_pzt, ti_pzt, te_pzt, tb_pzt_airflow, ti_pzt_airflow, te_pzt_airflow = time_compute(
            np.array(TP_pzt["exp"]), np.array(TP_pzt["insp"]), np.array(FN_pzt["exp"]), np.array(FN_pzt["insp"]), positives_pzt["exp"], positives_pzt["insp"])

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
                'TP_i': TP_mag["insp"],
                'FP_i': FP_mag["insp"],
                'FN_i': FN_mag["insp"],
                'TP_e': TP_mag["exp"],
                'FP_e': FP_mag["exp"],
                'FN_e': FN_mag["exp"],
                'tI (s)': ti_mag,
                'tE (s)': te_mag,
                'tB (s)': tb_mag,
                'tI airflow (s)': ti_mag_airflow,
                'tE airflow (s)': te_mag_airflow,
                'tB airflow (s)': tb_mag_airflow,
                'delay_i': delays_mag["insp"],
                'delay_e': delays_mag["exp"],
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
                'TP_i': TP_pzt["insp"],
                'FP_i': FP_pzt["insp"],
                'FN_i': FN_pzt["insp"],
                'TP_e': TP_pzt["exp"],
                'FP_e': FP_pzt["exp"],
                'FN_e': FN_pzt["exp"],
                'delay_i': delays_pzt["insp"],
                'delay_e': delays_pzt["exp"],
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
                'TP_i_indx': positives_pzt["insp"],
                'TP_e_indx': positives_pzt["exp"],
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
