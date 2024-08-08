# third-party
import scipy
import numpy as np


def get_template_markers(device_overview, max_len=0):

    markers = device_overview["peaks"] + device_overview["valleys"]
    markers.sort()

    templates_ind = []

    for i in range(0, len(markers)-1, 4):

        try:
            midpoint_start = int((markers[i+1] + markers[i]) / 2)
            midpoint_end = int((markers[i+3] + markers[i+2]) / 2)
            max_len = max(max_len, midpoint_end - midpoint_start)
            templates_ind += [[midpoint_start, midpoint_end]] 
        except:
            pass

    return templates_ind, max_len



def get_waveforms_by_id_by_activity(signal, device_overview, waveforms=None, max_len=0):
    ''' 
    Parameters
    ---------- 
    signal: array-like
        description

    device_overview: dict
        Overview of the device performance, including indexes for detected peaks and valleys
    
    Returns
    -------
    waveforms: 2D array
        Array containing the waveforms, resampled to the same length (first dimension is the number of waveforms, second dimension is the length of the waveforms)
    ''' 

    # signal -= np.mean(signal)
    templates_ind, max_len = get_template_markers(device_overview, max_len)
    
    if waveforms is None:
        waveforms = np.empty((0, max_len))

    for midpoint_start, midpoint_end in templates_ind:
        waveforms = np.append(waveforms, np.reshape(scipy.signal.resample(signal[midpoint_start : midpoint_end], max_len), (1, -1)), axis=0)

    return waveforms, max_len