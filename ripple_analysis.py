#Ripple detection
import numpy as np
from scipy.signal import butter, lfilter, hilbert
from ripple_detection import Kay_ripple_detector

#Filter functions
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#Filter LFP at ripple freq
def rip_filt_LFP(signal,data_TS,plot=True):
    lc = 150.0 #Low cut of filter
    hc = 200.0 #High cut of filter
    fs = 1000.0 #Sampling frequency (currently set at 1000Hz)
    
    #Take average of all membrane currents
    data = np.average(signal,axis=0)
    
    #Filtered signal
    filt_LFP = butter_bandpass_filter(data,lc,hc,fs,order=2)

    #Amplitude envelope
    analytic_signal = hilbert(filt_LFP)
    amplitude_envelope = np.abs(analytic_signal)
    
    if plot:
        plt.figure(figsize=(20,3))
        plt.plot(data_TS,filt_LFP)
        plt.plot(data_TS,amplitude_envelope)
        
    return filt_LFP, amplitude_envelope

#Run ripple detection to get ripple times
def rip_times(filt_LFP,data_TS,t_hold=1.5):  
    fs = 1000.0
    
    time = np.array(data_TS)
    speed = np.zeros(np.shape(filt_LFP))

    LFP = np.expand_dims(filt_LFP, axis=1)

    ripple_times = Kay_ripple_detector(time, LFP, speed, fs, zscore_threshold=t_hold)

    #Convert to numpy array
    return ripple_times.to_numpy()

#Generates arrays of amplitude and length for every ripple event
def rip_props(ripple_times, data_TS, amplitude_envelope):
    
    rip_amp = np.zeros(len(ripple_times))
    rip_length = np.zeros(len(ripple_times))
    for iRip in range(0,len(ripple_times)):
        rip_ind = np.where((data_TS >= ripple_times[iRip,0]) &
                 (data_TS <= ripple_times[iRip,1]))

        #Adds peak amplitude to amp array
        rip_amp[iRip] = np.max(amplitude_envelope[rip_ind[0][0]:rip_ind[0][-1]])

        #Adds ripple length to length array
        rip_length[iRip] = ripple_times[iRip,1]-ripple_times[iRip,0]
        
    return rip_amp, rip_length

#Calculates spike participation in ripple events
def rip_spike_part(ripple_times,data_Spikes,data_SpikeIndex,n_neurons=1000):
    
    #Gets time indices of every ripple
    for iRip in range(0,len(ripple_times)):

        #Finds every time index falling within a ripple event
        inds = np.where((data_Spikes >= ripple_times[iRip,0]) & 
                        (data_Spikes <= ripple_times[iRip,1]))

        #Adds all time indices falling within a ripple window to one super array
        if iRip == 0:
            ripple_time_indices = inds[0]
        else:
            ripple_time_indices = np.concatenate((ripple_time_indices,inds[0]))

    #Calculates number of ripple-associated spikes for each cell
    spikes_per_ripple = np.zeros((n_neurons,1))
    
    for iCell in range(0,n_neurons):
        
        cell_spikes = np.where(data_SpikeIndex == iCell)
        spikes_per_ripple[iCell] = len(np.intersect1d(cell_spikes, ripple_time_indices)) / len(ripple_times) 
        
    return spikes_per_ripple