# third-party
import numpy as np
from scipy.signal import find_peaks, peak_prominences, savgol_filter, detrend

def peak_valley(signal):
    peak, _ = find_peaks(signal, distance=100)#, height= np.ptp(integral) * 0.1)
    valley, _ = find_peaks(-signal, distance=100)#, height= np.ptp(integral) * 0.1)
    return np.asarray(peak), np.asarray(valley)


def detrend_signal(signal, sampling_rate=100, segment_length=10):
    # Calculate the number of samples in each 10-second segment
    segment_samples = int(sampling_rate * segment_length)

    # Calculate the number of segments
    num_segments = len(signal) // segment_samples

    # Detrend each segment separately
    detrended_signal = []
    for i in range(num_segments):
        start_idx = i * segment_samples
        end_idx = (i + 1) * segment_samples
        segment = signal[start_idx:end_idx]
        detrended_segment = detrend(segment)
        detrended_signal.extend(detrended_segment)

    if len(signal) % segment_samples != 0:
        start_idx = (i + 1) * segment_samples
        segment = signal[start_idx:]
        detrended_segment = detrend(segment)
        detrended_signal.extend(detrended_segment)

    # Convert the result back to a NumPy array
    detrended_signal = np.array(detrended_signal)

    return detrended_signal


def preprocess(mag_data, airflow_data, pzt_data):

    # de-mean
    mag_data = mag_data - mag_data.mean()
    airflow_data = airflow_data - airflow_data.mean()
    pzt_data = pzt_data - pzt_data.mean()

    # invert MAG data
    mag_data = -mag_data

    #filter
    mag_data = savgol_filter(mag_data, 100, 2)
    pzt_data = savgol_filter(pzt_data, 100, 2)
    
    #integral
    X = np.asarray(airflow_data - np.mean(airflow_data))
    dt = 1/100
    airflow_data = np.cumsum(X) * dt

    pzt_data = detrend_signal(pzt_data)
    Y = np.asarray(pzt_data - np.mean(pzt_data))
    pzt_data = np.cumsum(Y) * dt

    return mag_data, airflow_data, pzt_data



def remove_extrems(peaks_biopac, valley_biopac, extrems, signal): 
    '''
    Run all signal 


    Output: signal with consecutive peaks and valleys
    '''
    remove = []
    okay = True
    for i in range(len(extrems)-1):
        #2 peak
        if ((extrems[i] in peaks_biopac) and (extrems[i+1] in peaks_biopac)):
            # delete the one with less amplitude
            if abs(signal[extrems[i+1]]-signal[extrems[i]])<10: #common in apnea
                remove.append(extrems[i+1]) 
            elif signal[extrems[i+1]]>signal[extrems[i]]:
                remove.append(extrems[i])
            else:
                remove.append(extrems[i+1])
        #2 exp
        elif ((extrems[i] in valley_biopac) and (extrems[i+1] in valley_biopac)):
            # remove the minimum with max amplitude
            if abs(signal[extrems[i+1]]-signal[extrems[i]])<10: 
                remove.append(extrems[i+1])
            elif signal[extrems[i]]>signal[extrems[i+1]]:
                remove.append(extrems[i])
            else:
                remove.append(extrems[i+1])
    if len(remove)!=0:
        okay=False
    return okay, np.asarray(remove)


def flow_reversal(signal):
    '''
    input: signal
    Compute peaks and valleys
    All signals - If peaks are consecutive or valleys are consecutive remove
    Remove 1 peak or valley if the amplitude peak-valley is less than a treshold
    
    The result signal has peaks and valleys consecutively, and conditions of prominance and amplitude are satisfied

    output: peaks and valley; Peaks correspond to the expiration start; Valleys correspond to the inspiration start
    '''
    okay =  False
    peak, valley = peak_valley(signal)
    peak_prom, valley_prom, peak_i, valley_i = peak_prominences(signal,peak)[0], peak_prominences(-signal,valley)[0], peak, valley
    extrems = np.concatenate((peak,valley))
    extrems = np.sort(extrems)
    while not okay: 
        okay, remove = remove_extrems(peak, valley, extrems, signal)
        mask = np.isin(peak, remove, invert=True)
        peak = peak[mask]
        mask = np.isin(valley, remove, invert=True)
        valley = valley[mask]
        extrems = np.concatenate((peak,valley))
        extrems = np.sort(extrems)

        distance = np.diff(extrems)
        mean_amplitude = np.mean(abs(np.diff(signal[extrems])))
        d=0
        remove1 = []
        while d<len(distance) and len(remove1)==0: 
            if abs(signal[extrems[d]]-signal[extrems[d+1]])<mean_amplitude*0.3: 
                if extrems[d] in peak:
                    peak_prominance = peak_prom[np.where(peak_i == extrems[d])[0][0]]
                    valley_prominance = valley_prom[np.where(valley_i == extrems[d+1])[0][0]]
                    if peak_prominance>valley_prominance:
                        remove1.append(extrems[d+1])
                    else:
                        remove1.append(extrems[d])         
                else: 
                    peak_prominance = peak_prom[np.where(peak_i == extrems[d+1])[0][0]]
                    valley_prominance = valley_prom[np.where(valley_i == extrems[d])[0][0]]
                    if peak_prominance>valley_prominance:
                        remove1.append(extrems[d])
                    else:
                        remove1.append(extrems[d+1])

                okay = False
            d=d+1        
        mask = np.isin(peak, remove1, invert=True)

        peak = peak[mask]
        mask = np.isin(valley, remove1, invert=True)
        valley = valley[mask]
        extrems = np.concatenate((peak,valley))
        extrems = np.sort(extrems)
    return peak, valley



def time_compute(peak, valley):
    '''
    input: indices peak(exp) and valleys(insp)
    output: breathing period (secs); considering a cycle inspiration+expiration; Rejecting isolated expirations at the begining and inspirations at the end
    time from valley to valley == from inspiration to inspiration
    
    '''
    sampling_freq=100
    extrems = np.concatenate((peak,valley))
    extrems = np.sort(extrems)
    dif = np.diff(extrems)
    interval = []
    #duty_cycle
    ds=[]
    breath_time, time_insp, time_exp, ds = [], [],[], []
    if extrems[0] in peak: 
        #dif=exp-insp-exp=peak-valley-peak
        time_insp = dif[1::2]
        time_exp = dif[2::2]
        for c in range(len(time_insp)):
            try: 
                breath_time.append(time_insp[c]+time_exp[c+1])
                interval.append([valley[c],valley[c+1]])
                ds.append((time_insp[c])/(time_insp[c]+time_exp[c+1]))
            except: 
                pass
        time_exp = np.concatenate((dif[0],time_exp),axis=None)


    else:
        #dif=insp-exp=valley-peak
        time_insp = dif[0::2]
        time_exp = dif[1::2]
        for c in range(len(time_insp)):
            try: 
                breath_time.append(time_insp[c]+time_exp[c])
                interval.append([valley[c],valley[c+1]])
                ds.append((time_insp[c]*100)/(time_insp[c]+time_exp[c]))
            except: 
                pass
    

    
    return np.asarray(breath_time)/sampling_freq, time_insp/sampling_freq, time_exp/sampling_freq, interval, ds


def thresDistance_peaks(point_ref, breaths, interval):
    '''
    input: period of each breathing
    output: threshold distance considering the mean of 5 breaths (or less)
    '''
    if len(breaths)<5:
        return (np.mean(breaths)/2)*100
    
    result = np.ones(len(breaths))
    for i in range(4, len(breaths)): 
        result[i] = (np.mean(breaths[i-4:i])/2)*100

    for i in range(5): 
        result[i] = result[4]
    
    posi = 0 
    for i,j in interval: 
        #print('i,j', i,j, 'point_ref', point_ref)
        if point_ref>=i and point_ref<=j: 
            return result[posi]
        posi = posi + 1

    if abs(interval[0][0]-point_ref)<abs(interval[-1][0]-point_ref):
        return result[0]
    else:
        return result[-1]


#nao funcionou muito bem 
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



def evaluate_extremums(extrems, extrem_reference, breaths_reference, interval):
    '''
    input: extrems (inspiration=valleys or expiration=peaks); 
    extrem_reference (BIOPAC: inspiration=valleys or expiration=peaks);
    interval: interval of a breath cycle (inspiration and expiration);

    output: evaluate the extrems regarding the reference; 
    extrems can be classified as TP, FP, FN and delays are are computed
    '''
    extrem_reference = np.array(extrem_reference)
    performance_clf, positives  = {}, {}
    TP, FP, FN, delays = [], [], [], []
    #print('extrems', extrems, 'ref', extrem_reference)

    for extremum in extrems:
        insp_distance = np.abs(extrem_reference - extremum)
        closest_extrem_index = np.argmin(insp_distance)
        closest_extrem = extrem_reference[closest_extrem_index]
        thres_distance = thresDistance_peaks(closest_extrem, breaths_reference, interval)
        
        delay = closest_extrem - extremum
        performance_clf[extremum] = {}
        #print('extremum', extremum, 'closest_extrem', closest_extrem, 'delay', delay, 'ther', thres_distance)
        if closest_extrem in positives.keys():
            if abs(positives[closest_extrem]['delay']) > abs(delay):
            #change for the new point
                #print('closest_extrem',closest_extrem,'extremum',extremum,' abs(positives[closest_extrem][delay])', abs(positives[closest_extrem]['delay']), 'abs(delay)', abs(delay))
                TP.remove(positives[closest_extrem]['point'])
                FP.append(positives[closest_extrem]['point'])
                performance_clf[positives[closest_extrem]['point']]['closest'] = 'NA'
                performance_clf[positives[closest_extrem]['point']]['delay'] = 'NA'
                performance_clf[positives[closest_extrem]['point']]['clf'] = 'FP'
                positives[closest_extrem]['delay'] = delay
                positives[closest_extrem]['point'] = extremum
                
    
            #add new point 
                performance_clf[extremum]['closest'] = closest_extrem
                performance_clf[extremum]['delay'] = delay
                performance_clf[extremum]['clf'] = 'TP'
                TP.append(extremum)
            else: 
                
                sorted_indices = np.argsort(insp_distance)
                closest_extrem = extrem_reference[sorted_indices[1]]
                delay = insp_distance[sorted_indices[1]]
                thres_distance = thresDistance_peaks(closest_extrem, breaths_reference, interval)
                if delay < thres_distance:
                    #add new point
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
                #add
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
    #print('FP', FP, 'TP', TP, 'FN', FN, 'len(insp)', len(extrems), 'FP+TP', len(FP)+len(TP))
    #print(performance_clf)


    for p in performance_clf.keys():
        if performance_clf[p]['clf']=='TP':
            delays.append( performance_clf[p]['delay']) 
    
    return FP, TP, FN, performance_clf, positives, delays


def correct_variable(variable, variable_reference,array_TP,array_FP, array_FN):
    '''
    Pair-by-pair comparison implies that the variables computed are in the same number and have the rigth correspondence SENSOR and REFERENCE
    For the sensor only values corresponding to TP are considered
    For the reference undetected values by the sensor are removed
    
    Variables tI, tB depend on TP_i
    Variables tE depend on TP_e

    '''
    result_array = []
    result_variable = []
    positions = np.concatenate((array_TP,array_FP, array_FN))
    positions  = np.sort(positions)

    v,p=0,0
    while v<len(variable) and p<len(positions):
        if positions[p] in array_TP:
            result_array.append('TP')
            result_variable.append(variable[v])
            v=v+1
        elif positions[p] in array_FP:
            result_array.append('FP')
            v=v+1
        elif positions[p] in array_FN:
            result_array.append('FN')
            result_variable.append(None)
        p = p + 1

    #FN_positions = [index for index, value in enumerate(result_array) if value is None]
    variable_reference_final = [x for x, m in zip(variable_reference, result_variable) if m is not None]
    variable_final = [m for x, m in zip(variable_reference, result_variable) if m is not None]
    
    #result_variable = [x for x in result_variable if x is not None]
    #list(filter(lambda x: x is not None, result_variable))


    return variable_final, variable_reference_final
