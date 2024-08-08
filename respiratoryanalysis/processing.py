# third-party
import numpy as np
from scipy.signal import find_peaks, peak_prominences, savgol_filter, filtfilt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import biosppy.signals.tools as st

# biosppy.signals.tools.find_extrema


def select_by_peak_prominence(extrema, signal, prominence_ratio=0.02, sampling_freq=100):
    """Select peaks by distance.
    Parameters
    ----------
    peaks : array
        The peaks to be selected.
    amplitudes : array
        The amplitudes of the peaks.
    distance : int
        The minimum distance between peaks.
    Returns
    -------
    selected_peaks : array
        The selected peaks.
    selected_amplitudes : array
        The amplitudes of the selected peaks.
    """

    prominences_peaks = peak_prominences(
        signal, extrema, wlen=30*sampling_freq)[0]
    prominences_valleys = peak_prominences(-signal,
                                           extrema, wlen=30*sampling_freq)[0]

    pkpk_amp = np.amax(signal) - np.amin(signal)
    prominence_thr = pkpk_amp * prominence_ratio

    return extrema[prominences_peaks > prominence_thr], extrema[prominences_valleys > prominence_thr]


def peak_valley(signal):
    # , height= np.ptp(integral) * 0.1)
    peak, _ = find_peaks(signal, distance=100)
    # , height= np.ptp(integral) * 0.1)
    valley, _ = find_peaks(-signal, distance=100)
    return np.asarray(peak), np.asarray(valley)


def peak_valley_new(signal, sampling_freq=100):
    # distance between peaks defined as 0.6s, as proposed in "Validation of a Wearable Device and an Algorithm for Respiratory Monitoring during Exercise"
    peak, _ = find_peaks(signal, distance=0.6 *
                         sampling_freq, wlen=30*sampling_freq)
    valley, _ = find_peaks(-signal, distance=0.6 *
                           sampling_freq, wlen=30*sampling_freq)
    return np.sort(np.concatenate((peak, valley)))


def preprocess4visualization(mag_data, airflow_data, pzt_data, window_size=100):

    # de-mean
    mag_data = mag_data - mag_data.mean()
    airflow_data = airflow_data - airflow_data.mean()
    pzt_data = pzt_data - pzt_data.mean()

    # invert MAG data
    mag_data = -mag_data

    # integral
    X = np.asarray(airflow_data - np.mean(airflow_data))
    dt = 1/100
    airflow_data = np.cumsum(X) * dt

    # filter
    mag_data = savgol_filter(mag_data, window_size, 2)
    pzt_data = savgol_filter(pzt_data, window_size, 2)
    # airflow_data = savgol_filter(airflow_data, window_size, 2)

    return mag_data, airflow_data, pzt_data


def preprocess(mag_data, airflow_data, pzt_data, sampling_freq=100, fc=[0.01, 0.35]):

    # de-mean
    mag_data = mag_data - mag_data.mean()
    airflow_data = airflow_data - airflow_data.mean()
    pzt_data = pzt_data - pzt_data.mean()

    # invert MAG data
    mag_data = -mag_data

    # integral
    X = np.asarray(airflow_data - np.mean(airflow_data))
    dt = 1/100
    airflow_data = np.cumsum(X) * dt

    mag_data = st.filter_signal(mag_data, ftype='butter', band='bandpass',
                                order=2, frequency=fc, sampling_rate=sampling_freq)['signal']
    pzt_data = st.filter_signal(pzt_data, ftype='butter', band='bandpass',
                                order=2, frequency=fc, sampling_rate=sampling_freq)['signal']
    airflow_data = st.filter_signal(airflow_data, ftype='butter', band='bandpass',
                                    order=2, frequency=fc, sampling_rate=sampling_freq)['signal']

    return mag_data, airflow_data, pzt_data


def remove_extrems(peaks_biopac, valley_biopac, extrems, signal, ref_signal=None):
    '''
    Run all signal 


    Output: signal with consecutive peaks and valleys
    '''
    remove = []
    okay = True
    for i in range(len(extrems)-1):
        # 2 peak
        if ((extrems[i] in peaks_biopac) and (extrems[i+1] in peaks_biopac)):
            # delete the one with less amplitude
            if abs(signal[extrems[i+1]]-signal[extrems[i]]) < 10:  # common in apnea
                remove.append(extrems[i+1])
            elif signal[extrems[i+1]] > signal[extrems[i]]:
                remove.append(extrems[i])
            else:
                remove.append(extrems[i+1])
        # 2 exp
        elif ((extrems[i] in valley_biopac) and (extrems[i+1] in valley_biopac)):
            # remove the minimum with max amplitude
            if abs(signal[extrems[i+1]]-signal[extrems[i]]) < 10:
                remove.append(extrems[i+1])
            elif signal[extrems[i]] > signal[extrems[i+1]]:
                remove.append(extrems[i])
            else:
                remove.append(extrems[i+1])
    if len(remove) != 0:
        okay = False
    return okay, np.asarray(remove)


def flow_reversal_new(signal):

    extrema = peak_valley(signal)

    peaks, valleys = select_by_peak_prominence(extrema, signal)

    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # # plot sensor and airflow signals
    # fig.add_trace(go.Scatter(y=signal, name="signal"))
    # fig.add_trace(go.Scatter(x=extrema, y=signal[extrema], mode='markers'))
    # for ext in extrema_keep:
    #     fig.add_vline(x=ext)
    # fig.show()

    return None


def flow_reversal(signal):
    '''
    input: signal
    Compute peaks and valleys
    All signals - If peaks are consecutive or valleys are consecutive remove
    Remove 1 peak or valley if the amplitude peak-valley is less than a treshold

    The result signal has peaks and valleys consecutively, and conditions of prominance and amplitude are satisfied

    output: peaks and valley; Peaks correspond to the expiration start; Valleys correspond to the inspiration start
    '''
    okay = False
    peak, valley = peak_valley(signal)

    peak_prom, valley_prom, peak_i, valley_i = peak_prominences(
        signal, peak)[0], peak_prominences(-signal, valley)[0], peak, valley

    extrems = np.concatenate((peak, valley))
    extrems = np.sort(extrems)

    while not okay:
        okay, remove = remove_extrems(
            peak, valley, extrems, signal)
        mask = np.isin(peak, remove, invert=True)
        peak = peak[mask]
        mask = np.isin(valley, remove, invert=True)
        valley = valley[mask]
        extrems = np.concatenate((peak, valley))
        extrems = np.sort(extrems)

        distance = np.diff(extrems)
        mean_amplitude = np.mean(abs(np.diff(signal[extrems])))
        d = 0
        remove1 = []
        while d < len(distance) and len(remove1) == 0:
            if abs(signal[extrems[d]]-signal[extrems[d+1]]) < mean_amplitude*0.3:
                if extrems[d] in peak:
                    peak_prominance = peak_prom[np.where(
                        peak_i == extrems[d])[0][0]]
                    valley_prominance = valley_prom[np.where(
                        valley_i == extrems[d+1])[0][0]]
                    if peak_prominance > valley_prominance:
                        remove1.append(extrems[d+1])
                    else:
                        remove1.append(extrems[d])
                else:
                    peak_prominance = peak_prom[np.where(
                        peak_i == extrems[d+1])[0][0]]
                    valley_prominance = valley_prom[np.where(
                        valley_i == extrems[d])[0][0]]
                    if peak_prominance > valley_prominance:
                        remove1.append(extrems[d])
                    else:
                        remove1.append(extrems[d+1])

                okay = False
            d = d+1
        mask = np.isin(peak, remove1, invert=True)

        peak = peak[mask]
        mask = np.isin(valley, remove1, invert=True)
        valley = valley[mask]
        extrems = np.concatenate((peak, valley))
        extrems = np.sort(extrems)
    return peak, valley


def get_TP_times(extrema, extrema_ref_og, mode, first_is_peak=None):
    """
    extrema: list of tuples (index of extremum, TP/FN (True/False), peak/valley)
    """

    extrema_ref = extrema_ref_og.copy()

    for ind in np.where(np.array([a[1] for a in extrema]) == False)[0]:
        extrema_ref.insert(ind, (0, 0, 0))

    # extrema_pairs: list of tuples (index of extremum, index of next extremum, if both extrema TP (True/False))
    # extrema_pairs_ref: list of tuples (index of extremum, index of next extremum)

    if mode == "exp":
        first_event = "peak"
        second_event = "valley"
        jump = 1
    elif mode == "insp":
        first_event = "valley"
        second_event = "peak"
        jump = 1
    else:  # mode == "tB"
        if first_is_peak:
            first_event = "peak"
            second_event = "peak"
        else:
            first_event = "valley"
            second_event = "valley"
        jump = 2

    extrema_pairs = [
        (extrema[i][0], extrema[i+jump][0],
            all((extrema[i][1], extrema[i+jump][1])))
        for i in range(0, len(extrema)-jump)
        if extrema[i][2] == first_event and extrema[i+jump][2] == second_event
    ]

    # extrema_pairs_ref = [
    #     (extrema_ref[i][1], extrema_ref[i+jump][1])
    #     for i in range(0, len(extrema)-jump)
    #     if extrema[i][2] == first_event and extrema[i+jump][2] == second_event
    # ]

    extrema_pairs_ref = [
        (list(filter(lambda x: x[0] == pair[0], extrema_ref))[0][1], list(
            filter(lambda x: x[0] == pair[1], extrema_ref))[0][1])
        if list(filter(lambda x: x[0] == pair[0], extrema_ref)) != [] and list(filter(lambda x: x[0] == pair[1], extrema_ref)) != []
        else (0, 0)
        for pair in extrema_pairs
    ]

    extrema_pairs_both = np.hstack((extrema_pairs_ref, extrema_pairs))
    TP_extrema_pairs = extrema_pairs_both[extrema_pairs_both[:, -1] == True]

    return np.diff(TP_extrema_pairs[:, 2:4]).flatten(), np.diff(TP_extrema_pairs[:, 0:2]).flatten()


def time_compute(TP_peaks, TP_valleys, FN_peaks, FN_valleys, ref_peaks, ref_valleys, sampling_freq=100):

    # create tuples with the indices of the events and a boolean indicating if it is a True event (TP) or False event (FN)
    TP_peaks_tuple = [(event, True, "peak") for event in TP_peaks]
    TP_valleys_tuple = [(event, True, "valley") for event in TP_valleys]
    FN_peaks_tuple = [(event, False, "peak") for event in FN_peaks]
    FN_valleys_tuple = [(event, False, "valley") for event in FN_valleys]

    ref_peaks_tuple = [(ref_peaks[ref_ind]["point"], ref_ind, "peak")
                       for ref_ind in ref_peaks.keys()]
    ref_valleys_tuple = [(ref_valleys[ref_ind]["point"], ref_ind, "valley")
                         for ref_ind in ref_valleys.keys()]

    # concatenate and sort all events
    extrema = sorted(TP_peaks_tuple + TP_valleys_tuple +
                     FN_peaks_tuple + FN_valleys_tuple)
    extrema_ref = sorted(ref_peaks_tuple + ref_valleys_tuple)

    # get expiration times (tE) -> need to make sure sequence starts at peak
    time_exp, time_exp_ref = get_TP_times(
        extrema, extrema_ref,
        # extrema[0][0] in TP_peaks or extrema[0][0] in FN_peaks,
        mode="exp")
    # get inspiration times (tI) -> need to make sure sequence starts at valley
    time_insp, time_insp_ref = get_TP_times(
        extrema, extrema_ref,
        # extrema[0][0] in TP_valleys or extrema[0][0] in FN_valleys,
        mode="insp")

    # get breathing period (tB)
    # valleys = sorted(TP_valleys_tuple + FN_valleys_tuple)
    breath_time, breath_time_ref = get_TP_times(
        extrema, extrema_ref, mode="tB",
        first_is_peak=extrema[0][0] in TP_peaks or extrema[0][0] in FN_peaks)

    tB = np.asarray(breath_time)/sampling_freq
    tI = time_insp/sampling_freq
    tE = time_exp/sampling_freq

    tB_ref = np.asarray(breath_time_ref)/sampling_freq
    tI_ref = time_insp_ref/sampling_freq
    tE_ref = time_exp_ref/sampling_freq

    return tB, tI, tE, tB_ref, tI_ref, tE_ref


def time_compute_prev(peak, valley, FP_peak=None, FP_valley=None, positives_peak_ref=0, positives_valley_ref=0):
    '''
    input: indices peak(exp) and valleys(insp)
    output: breathing period (secs); considering a cycle inspiration+expiration; Rejecting isolated expirations at the begining and inspirations at the end
    time from valley to valley == from inspiration to inspiration

    '''
    if FP_peak is not None and len(FP_peak) != 0:
        print("here")
    if (len(peak) == 0 or len(valley) == 0) and positives_peak_ref == 0:
        return np.array([]), np.array([]), np.array([]), [], []
    elif (len(peak) == 0 or len(valley) == 0) and positives_peak_ref != 0:
        return np.array([]), np.array([]), np.array([]), [], [], np.array([]), np.array([]), np.array([]), []

    sampling_freq = 100
    extrems = np.concatenate((peak, valley))
    extrems = np.sort(extrems)
    dif = np.diff(extrems)
    interval = []

    if positives_peak_ref != 0:
        peak_ref = np.array(list(positives_peak_ref.keys()))
        valley_ref = np.array(list(positives_valley_ref.keys()))
        extrems_ref = np.sort(np.concatenate((peak_ref, valley_ref)))
        dif_ref = np.diff(extrems_ref)

    # duty_cycle
    ds = []
    breath_time, time_insp, time_exp, ds = [], [], [], []
    breath_time_ref, time_insp_ref, time_exp_ref, ds_ref = [], [], [], []
    if extrems[0] in peak:
        # dif=exp-insp-exp=peak-valley-peak
        time_insp = dif[1::2]
        time_exp = dif[2::2]

        if positives_peak_ref != 0:
            time_insp_ref = dif_ref[1::2]
            time_exp_ref = dif_ref[2::2]

        for c in range(len(time_insp)):
            try:
                if positives_peak_ref != 0:
                    breath_time_ref.append(
                        time_insp_ref[c] + time_exp_ref[c])
                    ds_ref.append(
                        (time_insp_ref[c]) / (time_insp_ref[c] + time_exp_ref[c]))

                breath_time.append(time_insp[c] + time_exp[c])
                interval.append([valley[c], valley[c + 1]])
                ds.append((time_insp[c]) / (time_insp[c] + time_exp[c]))

            except:
                pass
        try:  # catches when there are no complete breaths
            time_exp = np.concatenate((dif[0], time_exp), axis=None)

            if positives_peak_ref != 0:
                time_exp_ref = np.concatenate(
                    (dif_ref[0], time_exp_ref), axis=None)

        except Exception as e:
            print(e)
            pass

    else:
        # dif=insp-exp=valley-peak
        time_insp = dif[0::2]
        time_exp = dif[1::2]
        if positives_peak_ref != 0:
            time_insp_ref = dif_ref[0::2]
            time_exp_ref = dif_ref[1::2]

        for c in range(len(time_insp)):
            try:
                if positives_peak_ref != 0:
                    breath_time_ref.append(time_insp_ref[c]+time_exp_ref[c])
                    ds_ref.append(
                        (time_insp_ref[c] * 100) / (time_insp_ref[c] + time_exp_ref[c]))

                breath_time.append(time_insp[c] + time_exp[c])
                interval.append([valley[c], valley[c + 1]])
                ds.append((time_insp[c]*100) / (time_insp[c] + time_exp[c]))

            except:
                pass

    tB = np.asarray(breath_time)/sampling_freq
    tI = time_insp/sampling_freq
    tE = time_exp/sampling_freq

    if positives_peak_ref != 0:
        tB_ref = np.asarray(breath_time_ref)/sampling_freq
        tI_ref = time_insp_ref/sampling_freq
        tE_ref = time_exp_ref/sampling_freq

        if len(tB) != len(tB_ref) or len(tI) != len(tI_ref) or len(tE) != len(tE_ref):
            print('len(tB)', len(tB), 'len(tB_ref)', len(tB_ref), 'len(tI)',
                  len(tI), 'len(tI_ref)', len(tI_ref), 'len(tE)', len(tE), 'len(tE_ref)', len(tE_ref))

        return tB, tI, tE, interval, ds, tB_ref, tI_ref, tE_ref, ds_ref
    else:
        return tB, tI, tE, interval, ds


def thresDistance_peaks(point_ref, breaths, interval, thrs):
    '''
    input: period of each breathing
    output: threshold distance considering the mean of 5 breaths (or less)
    '''
    if len(breaths) < 5:
        return (np.mean(breaths)*thrs)*100

    result = np.ones(len(breaths))
    for i in range(4, len(breaths)):
        result[i] = (np.mean(breaths[i-4:i])*thrs)*100

    for i in range(5):
        result[i] = result[4]

    posi = 0
    for i, j in interval:
        # print('i,j', i,j, 'point_ref', point_ref)
        if point_ref >= i and point_ref <= j:
            return result[posi]
        posi = posi + 1

    if abs(interval[0][0]-point_ref) < abs(interval[-1][0]-point_ref):
        return result[0]
    else:
        return result[-1]


# nao funcionou muito bem
def compute_snr(signal):
    sampling_rate = 100  # Replace with your actual sampling rate
    piece_duration = 10  # Duration of each piece in seconds

    # Calculate the number of divisions (M)
    M = len(signal) // (piece_duration * sampling_rate)

    # Divide the signal into M pieces
    pieces = np.array_split(signal[:M * piece_duration * sampling_rate], M)

    # Create the signal matrix X
    X = np.vstack(pieces)

    # Compute the autocorrelation matrix D
    D = np.dot(X, X.T)

    # Calculate the eigenvalues of D
    eigenvalues = np.linalg.eigvalsh(D)

    # Extract the largest eigenvalue (λ1) and the sum of the remaining eigenvalues (λ2 + λ3 + ... + λM)
    lambda_1 = eigenvalues[-1]
    lambda_remaining = np.sum(eigenvalues[:-1])

    # Add epsilon to the denominator to avoid division by zero

    # Compute the SNR in decibels
    return 10 * np.log10(lambda_1 / lambda_remaining)


def evaluate_extrema(peaks, peaks_ref, valleys, valleys_ref, tb_ref, interval_ref, threshold_acceptability):

    TP, FP, FN = {'exp': [], 'insp': []}, {
        'exp': [], 'insp': []}, {'exp': [], 'insp': []}
    positives = {'exp': {}, 'insp': {}}
    delays = {'exp': [], 'insp': []}

    peaks_ref_tuple = [(event, "peak") for event in peaks_ref]
    valleys_ref_tuple = [(event, "valley") for event in valleys_ref]
    extrema_ref_tuple = sorted(peaks_ref_tuple + valleys_ref_tuple)

    last_event_found_ind = 0

    for extremum_ref in extrema_ref_tuple:
        if extremum_ref[1] == "peak":
            candidate_events = peaks[peaks > last_event_found_ind]
            event = "exp"
        else:
            candidate_events = valleys[valleys > last_event_found_ind]
            event = "insp"

        if len(candidate_events) == 0:
            FN[event].append(extremum_ref[0])
            break

        extremum_distance = np.abs(
            candidate_events - extremum_ref[0]
        )
        closest_extrem_index = np.argmin(extremum_distance)
        closest_extrem = candidate_events[closest_extrem_index]
        thres_distance = thresDistance_peaks(
            closest_extrem, tb_ref, interval_ref, threshold_acceptability)

        delay = extremum_ref[0] - closest_extrem

        if np.abs(delay) < thres_distance:
            positives[event][extremum_ref[0]] = {}
            TP[event].append(closest_extrem)
            delays[event].append(delay)
            positives[event][extremum_ref[0]]['point'] = closest_extrem
            positives[event][extremum_ref[0]]['delay'] = delay
            last_event_found_ind = closest_extrem

        else:
            FN[event].append(extremum_ref[0])

    for peak in peaks:
        if peak not in TP["exp"]:
            FP["exp"].append(peak)

    for valley in valleys:
        if valley not in TP["insp"]:
            FP["insp"].append(valley)

    return FP, TP, FN, positives, delays


def evaluate_extremums_prev(extrems, extrem_reference, breaths_reference, interval):
    '''
    input: extrems (inspiration=valleys or expiration=peaks); 
    extrem_reference (BIOPAC: inspiration=valleys or expiration=peaks);
    interval: interval of a breath cycle (inspiration and expiration);

    output: evaluate the extrems regarding the reference; 
    extrems can be classified as TP, FP, FN and delays are are computed
    '''
    extrem_reference = np.array(extrem_reference)
    performance_clf, positives = {}, {}
    TP, FP, FN, delays = [], [], [], []

    for extremum in extrems:
        insp_distance = np.abs(extrem_reference - extremum)
        closest_extrem_index = np.argmin(insp_distance)
        closest_extrem = extrem_reference[closest_extrem_index]
        thres_distance = thresDistance_peaks(
            closest_extrem, breaths_reference, interval)

        delay = closest_extrem - extremum
        performance_clf[extremum] = {}
        # print('extremum', extremum, 'closest_extrem', closest_extrem, 'delay', delay, 'ther', thres_distance)
        if closest_extrem in positives.keys():
            if abs(positives[closest_extrem]['delay']) > abs(delay):
                # change for the new point
                # print('closest_extrem',closest_extrem,'extremum',extremum,' abs(positives[closest_extrem][delay])', abs(positives[closest_extrem]['delay']), 'abs(delay)', abs(delay))
                TP.remove(positives[closest_extrem]['point'])
                FP.append(positives[closest_extrem]['point'])
                performance_clf[positives[closest_extrem]
                                ['point']]['closest'] = 'NA'
                performance_clf[positives[closest_extrem]
                                ['point']]['delay'] = 'NA'
                performance_clf[positives[closest_extrem]
                                ['point']]['clf'] = 'FP'
                positives[closest_extrem]['delay'] = delay
                positives[closest_extrem]['point'] = extremum

            # add new point
                performance_clf[extremum]['closest'] = closest_extrem
                performance_clf[extremum]['delay'] = delay
                performance_clf[extremum]['clf'] = 'TP'
                TP.append(extremum)
            else:

                sorted_indices = np.argsort(insp_distance)
                closest_extrem = extrem_reference[sorted_indices[1]]
                delay = insp_distance[sorted_indices[1]]
                thres_distance = thresDistance_peaks(
                    closest_extrem, breaths_reference, interval)
                if delay < thres_distance:
                    # add new point
                    performance_clf[extremum]['closest'] = closest_extrem
                    performance_clf[extremum]['delay'] = delay
                    performance_clf[extremum]['clf'] = 'TP'
                    positives[closest_extrem] = {}
                    positives[closest_extrem]['point'] = extremum
                    positives[closest_extrem]['delay'] = delay
                    TP.append(extremum)
                else:
                    performance_clf[extremum] = {}
                    performance_clf[extremum]['clf'] = 'FP'
                    FP.append(extremum)

        else:
            if delay < thres_distance:
                # add
                positives[closest_extrem] = {}
                performance_clf[extremum]['closest'] = closest_extrem
                performance_clf[extremum]['clf'] = 'TP'
                TP.append(extremum)
                performance_clf[extremum]['delay'] = delay
                positives[closest_extrem]['point'] = extremum
                positives[closest_extrem]['delay'] = delay

            else:
                performance_clf[extremum] = {}
                performance_clf[extremum]['clf'] = 'FP'
                FP.append(extremum)

    for ref in extrem_reference:
        if ref not in positives.keys():
            FN.append(ref)
    # print('FP', FP, 'TP', TP, 'FN', FN, 'len(insp)', len(extrems), 'FP+TP', len(FP)+len(TP))
    # print(performance_clf)

    for p in performance_clf.keys():
        if performance_clf[p]['clf'] == 'TP':
            delays.append(performance_clf[p]['delay'])

    return FP, TP, FN, performance_clf, positives, delays


def correct_variable(variable, variable_reference, array_TP, array_FP, array_FN):
    '''
    Pair-by-pair comparison implies that the variables computed are in the same number and have the rigth correspondence SENSOR and REFERENCE
    For the sensor only values corresponding to TP are considered
    For the reference undetected values by the sensor are removed

    Variables tI, tB depend on TP_i
    Variables tE depend on TP_e

    '''
    result_array = []
    result_variable = []
    positions = np.concatenate((array_TP, array_FP, array_FN))
    positions = np.sort(positions)

    v, p = 0, 0
    while v < len(variable) and p < len(positions):
        if positions[p] in array_TP:
            result_array.append('TP')
            result_variable.append(variable[v])
            v = v+1
        elif positions[p] in array_FP:
            result_array.append('FP')
            v = v+1
        elif positions[p] in array_FN:
            result_array.append('FN')
            result_variable.append(None)
        p = p + 1

    # FN_positions = [index for index, value in enumerate(result_array) if value is None]
    variable_reference_final = [x for x, m in zip(
        variable_reference, result_variable) if m is not None]
    variable_final = [m for x, m in zip(
        variable_reference, result_variable) if m is not None]

    # result_variable = [x for x in result_variable if x is not None]
    # list(filter(lambda x: x is not None, result_variable))

    return variable_final, variable_reference_final
