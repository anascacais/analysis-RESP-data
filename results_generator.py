# third-party 
import numpy as np
from plotly import graph_objs as go

# local
from files import save_results
from processing import preprocess, flow_reversal, time_compute, evaluate_extremums, compute_snr

def write_results(id, scientisst_data, biopac_data, bitalino_data, activities_info, show_fig=False):
    
    participant_results = {}
    
    for activity in activities_info.keys():

        if show_fig:
            fig = go.Figure()

        a = scientisst_data['RESP'][activities_info[activity]['start_ind_scientisst'] : activities_info[activity]['start_ind_scientisst'] + activities_info[activity]['length']]
        b = biopac_data['airflow'][activities_info[activity]['start_ind_biopac'] : activities_info[activity]['start_ind_biopac'] + activities_info[activity]['length']]
        c = bitalino_data['PZT'][activities_info[activity]['start_ind_bitalino'] : activities_info[activity]['start_ind_bitalino'] + activities_info[activity]['length']]

        N = len(b)
        time_x = np.linspace(0, N, N)

        a, b, c, integral = preprocess(a, b, c)

        # plot
        if show_fig:
            fig.add_trace(go.Scatter(y = a * 10, mode = 'lines', name='ScientISST'))
            fig.add_trace(go.Scatter(y = integral, mode = 'lines', name='BIOPAC_integral')) 
            fig.add_trace(go.Scatter(y = b, mode = 'lines', name='BIOPAC')) 
            fig.add_trace(go.Scatter(y = c, mode = 'lines', name='BITalino'))

        peaks_a, valleys_a = flow_reversal(a)
        peaks_b, valleys_b = flow_reversal(integral)
        peaks_c, valleys_c = flow_reversal(c)

        tb_b, ti_b , te_b, interval_b, ds_b = time_compute(peaks_b, valleys_b)
        br_b = (60 * len(tb_b)) / np.sum(tb_b)

        # evaluate peaks and valleys from ScientISST
        FP_s_e, TP_s_e, FN_s_e, performance_clf_s_e, positives_s_e, delay_s_e = evaluate_extremums(peaks_a, peaks_b, tb_b, interval_b)
        FP_s_i, TP_s_i, FN_s_i, performance_clf_s_i, _, delay_s_i = evaluate_extremums(valleys_a, valleys_b, tb_b, interval_b)
        # evaluate peaks and valleys from BITalino
        FP_c_e, TP_c_e, FN_c_e, _, _, delay_c_e = evaluate_extremums(peaks_c, peaks_b, tb_b, interval_b)
        FP_c_i, TP_c_i, FN_c_i, _, _, delay_c_i = evaluate_extremums(valleys_c, valleys_b, tb_b, interval_b)


        tb_a, ti_a , te_a, _, ds_a = time_compute(peaks_a, valleys_a)
        tb_c, ti_c , te_c, _, ds_c = time_compute(peaks_c, valleys_c)


        print('Scientisst: tb', tb_a, 'ti' ,ti_a ,'te', te_a)
        br_a = (60*len(tb_a))/np.sum(tb_a)
        br_c = (60*len(tb_c))/np.sum(tb_c)
        brv_a = (np.std(tb_a) / np.mean(tb_a)) * 100
        brv_b = (np.std(tb_b) / np.mean(tb_b)) * 100
        brv_c = (np.std(tb_c) / np.mean(tb_c)) * 100

        if show_fig:
            for i in peaks_a: 
                name = performance_clf_s_e[i]['clf']
                fig.add_vline(x=time_x[i], line_width=1, line_dash="dash", line_color="blue", name="", annotation_text=name)


            for i in valleys_a: 
                name = performance_clf_s_i[i]['clf']
                fig.add_vline(x=time_x[i], line_width=1, line_dash="dash", line_color="green", name="", annotation_text=name)


            for j in peaks_b: 
                if j not in positives_s_e.keys():
                    name = 'FN'
                else: 
                    name = ''
                fig.add_vline(x=time_x[j], line_width=1, line_dash="dash", line_color="black", name="", annotation_text=name)

        
        #flow_reversal
        #ratio_bitalino = (len(ti_c)+len(te_c))/(len(ti_b)+len(te_b))
        print((len(ti_a)+len(te_a))/(len(ti_b)+len(te_b)))
        if show_fig:
            fig.update_layout(title_text = activity)
            fig.show() 

        #save results:
        participant_results_activity = {
            'ScientISST':{
                'peaks': peaks_a.tolist(),
                'valleys': valleys_a.tolist(),
                'amplitude peaks': np.array(a)[peaks_a],
                'amplitude valleys': np.array(a)[valleys_a],
                'SNR': compute_snr(a),
                'TP_i': TP_s_i,
                'FP_i': FP_s_i,
                'FN_i': FN_s_i,
                'TP_e': TP_s_e,
                'FP_e': FP_s_e,
                'FN_e': FN_s_e,
                'tI (s)': ti_a,
                'tE (s)': te_a,
                'tB (s)': tb_a, 
                'delay_i': delay_s_i,
                'delay_e': delay_s_e,
                'ratio': (len(peaks_a) + len(valleys_a)) / (len(peaks_b) + len(valleys_b)),
                'ds (%)': ds_a,
                'BR (bpm)': br_a,
                'BRV (%)' : brv_a,
            },
            'BITalino':{
                'peaks': peaks_c.tolist(),
                'valleys': valleys_c.tolist(),
                'amplitude peaks':np.array(c)[peaks_c],
                'amplitude valleys':np.array(c)[valleys_c],
                'SNR': compute_snr(c),
                'TP_i': TP_c_i,
                'FP_i': FP_c_i,
                'FN_i': FN_c_i,
                'TP_e': TP_c_e,
                'FP_e': FP_c_e,
                'FN_e': FN_c_e,
                'delay_i': delay_c_i,
                'delay_e': delay_c_e,
                'tI (s)': ti_c,
                'tE (s)': te_c,
                'tB (s)': tb_c, 
                'ratio': (len(peaks_c) + len(valleys_c)) / (len(peaks_b) + len(valleys_b)),
                'BR (bpm)': br_c,
                'BRV (%)' : brv_c,
                'ds (%)': ds_c,
            },
            'BIOPAC':{
                'peaks_biopac': peaks_b.tolist(),
                'valleys_biopac': valleys_b.tolist(),
                'amplitude peaks':np.array(b)[peaks_b],
                'amplitude valleys':np.array(b)[valleys_b],
                'SNR': compute_snr(b),
                'tI (s)': ti_b.tolist(),
                'tE (s)': te_b.tolist(),
                'tB (s)': tb_b, 
                'BRV (%)' : brv_b,
                'ds (%)': ds_b,
                'BR (bpm)': br_b,
            }
        }


        participant_results[activity] = participant_results_activity

    save_results(participant_results, id)