import numpy as np
from scipy import signal
from tqdm import tqdm
from scipy.signal import medfilt
from scipy.signal import butter, sosfilt, sosfreqz, sosfiltfilt
from pyecg.data_handling import *



def remove_baseline(signal, fs=360):
	'''
	Applies two median filters to remove the baseline wander from the signal
	inputs:
			signal: a 1d numpy array
			fs:sampling frequency 
	output: a 1d array having the same shape as the input signal
	'''
	ker_size = int(0.2*fs)
	if ker_size % 2 == 0: ker_size += 1
	baseline = medfilt(signal, kernel_size=ker_size)
	
	ker_size = int(0.6*fs)
	if ker_siz % 2 == 0: ker_size += 2
	baseline = medfilt(baseline, kernel_size=ker_size)
	modified_signal = signal - baseline
	
	return modified_signal


def lowpass_filter_butter(signal, cutoff=45, fs=360,order=15):
    nyq = 0.5*fs
    cf = cutoff/nyq
    sos = butter(order, cf, btype='low', output='sos', analog=False)
    sig = sosfiltfilt(sos, signal) 
    return sig,sos 

def denoise_signal(signal,remove_bl=True,lowpass=False,fs=360,cutoff=45,order=15):
	'''
	inputs:
			signal: a 1d numpy array
	output: a 1d array having the same shape as the input signal
	'''	
	if remove_bl and not lowpass:
		y = remove_baseline(signal,fs=fs)
	if lowpass and not remove_bl:
		y,_ = lowpass_filter_butter(signal, cutoff=cutoff, fs=fs,order=order)
	if remove_bl and lowpass:
		y = remove_baseline(signal)
		y,_ = lowpass_filter_butter(y, cutoff=cutoff, fs=fs,order=order)
	if not remove_bl and not lowpass:
		y = signal

	return y

def plot_freq_response(sos,fs=360):

    w, h = sosfreqz(sos, worN=2000)
    plt.subplot(2, 1, 1)
    db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(w/np.pi*fs*0.5, db)
    plt.ylim(-75, 5)
    plt.grid(True)
    plt.yticks([0, -20, -40, -60])
    plt.ylabel('Gain [dB]')
    plt.title('Frequency Response')
    plt.subplot(2, 1, 2)
    plt.plot(w/np.pi*fs*0.5, np.angle(h))
    plt.grid(True)
    plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency (Hz)')
    plt.show()

def clean_inf_nan(ds):
    yds = ds['labels']
    xds = ds['waveforms']
    rds = ds['beat_feats']
    indexes=[]
    indexes.extend(np.where(np.isinf(rds))[0])
    indexes.extend(np.where(np.isnan(rds))[0])
    rds = np.delete(rds, indexes, axis=0)
    xds = np.delete(xds, indexes, axis=0)
    yds = np.delete(yds, indexes, axis=0)
    #ydsc = [it for ind,it in enumerate(yds) if ind not in indexes]
    
    return {'waveforms':xds,'beat_feats':rds,'labels':yds}

def clean_IQR(ds, factor=1.5, return_indexes=False):
    yds = ds['labels']
    xds = ds['waveforms']
    rds = ds['beat_feats']
    
    #clean a 2d array. Each column is a features, rows are samples. Only r.
    ind_outliers = []
    for i in range(rds.shape[1]):
        x = rds[:,i]
        Q1 = np.quantile(x, 0.25)
        Q3 = np.quantile(x, 0.75)
        IQR = Q3-Q1
        inds = np.where((x> (Q3+factor*IQR)) | (x< (Q1-factor*IQR)))[0]
        #print(len(inds))
        ind_outliers.extend(inds)
        
    rds = np.delete(rds, ind_outliers, axis=0)
    xds = np.delete(xds, ind_outliers, axis=0)
    yds = np.delete(yds, ind_outliers, axis=0)
    if return_indexes==False:
    	return {'waveforms':xds, 'beat_feats':rds, 'labels':yds}
    else:
    	return ind_outliers

def append_ds(ds1,ds2):
    dss=dict()
    dss['waveforms'] = np.vstack((ds1['waveforms'],ds2['waveforms']))
    dss['beat_feats'] = np.vstack((ds1['beat_feats'],ds2['beat_feats']))
    dss['labels'] = np.vstack((ds1['labels'].reshape(-1,1),ds2['labels'].reshape(-1,1))).flatten()
    return dss

def clean_IQR_class(ds,factor=1.5):
	#clean by IQR for every class separately
    for label in list(np.unique(ds['labels'])):
        sliced = slice_data(ds, [label])
        cleaned = clean_IQR(sliced, factor=factor)
        try:
            ds_all=append_ds(ds_all,cleaned)
        except NameError:
            ds_all=cleaned
    return ds_all



class STFT:
	"""
	Preprocesses raw signals 
	Example:
		dpr = STFT()
		features_train = dpr.specgram(x_train, Fs=360, nperseg=127, noverlap=122)
	"""
	def __init__(self):
		pass
	def specgram(self, signals, Fs=None, nperseg=None, noverlap=None):
		"""
		input: 2d array of raw signals
		output: 3d array of transformed signals
		"""
		if Fs==None:
			Fs=360
		if nperseg == None:
			nperseg=64
		if noverlap == None:
			noverlap = int(nperseg/2)
		list_all=[]
		for i in tqdm(range(len(signals))):
			f,t,Sxx= signal.spectrogram(signals[i], fs=Fs, nperseg=nperseg, noverlap=noverlap, mode='psd')
			list_all.append(Sxx.T[:,:].tolist())
		out = np.array(list_all)
		return out
	
	def calc_feat_dim(self,samp,win,overlap):
		import math
		hdim = math.floor((samp-overlap)/(win-overlap))
		vdim = math.floor(win/2+1)
		return hdim,vdim 


