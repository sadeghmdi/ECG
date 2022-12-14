


import numpy as np
from scipy import stats





def get_stat_features(data, features='all'):
    """
    Computes statistical features for the input samples.

    Parameters
    ----------
    data : 2D numpy.array
        A 2D numpy array with shape (#samples,len_series).
    features : list
        A list of features to be computed. 

    Returns
    -------
    features_arr : 2D numpy.array
        A 2D numpy array with the shape (#samples, #features).
    """
    if features == 'all':
        flist = ['max','min','mean','std',
                         'median','skew','kurtosis',
                         'range']
    else:
        flist = features 

    num_samples = data.shape[0]
    num_features = len(flist)
    features_arr = np.ndarray((num_samples,num_features),dtype=np.float32)

    if 'max' in flist:
        ix = flist.index('max')
        features_arr[:,ix] = np.amax(data,axis=1)

    if 'min' in flist:
        ix = flist.index('min')
        features_arr[:,ix] = np.amin(data,axis=1)

    if 'mean' in flist:
        ix = flist.index('mean')
        features_arr[:,ix] = np.mean(data,axis=1)
    
    if 'std' in flist:
        ix = flist.index('std')
        features_arr[:,ix] = np.std(data,axis=1)
    
    if 'median' in flist:
        ix = flist.index('median')
        features_arr[:,ix] = np.median(data,axis=1)
    
    if 'skew' in flist:
        ix = flist.index('skew')
        features_arr[:,ix] = stats.skew(data,axis=1)
    
    if 'kurtosis' in flist:
        ix = flist.index('kurtosis')
        features_arr[:,ix] = stats.kurtosis(data,axis=1)
    
    if 'range' in flist:
        ix = flist.index('range')
        features_arr[:,ix] = np.amax(data,axis=1)-np.amin(data,axis=1)

    return features_arr



def get_hrv_features(rri, features='all'):
    """
    Computes hrv features for the input samples.

    Parameters
    ----------
    rri : 2D numpy.array
        A 2D numpy array with shape (#samples,len_series).
        Series are rr intervals in miliseconds(ms)
    features : list
        A list of features to be computed. 

    Returns
    -------
    features_arr : 2D numpy.array
        A 2D numpy array with the shape (#samples, #features).
    """
    if features == 'all':
        flist = ['meanrr','sdrr','medianrr','rangerr','nsdrr','sdsd','rmssd','nrmssd']
        flist += ['prr50']
        #flist += ['meanhr','maxhr','minhr','medianhr','sdhr']

    else:
        flist = features #features list

    num_samples = rri.shape[0]
    num_features = len(flist)
    features_arr = np.ndarray((num_samples,num_features),dtype=np.float32)

    #successive RR interval differences
    sd = np.diff(rri, axis=1)

    #calculate meanrr
    if bool({'meanrr','nsdrr','nrmssd'} & set(flist)):
        meanrr = np.mean(rri,axis=1)

    #calculate meanrr
    if bool({'sdrr','nsdrr'} & set(flist)):
        sdrr = np.std(rri,axis=1)

    #calculate rmssd
    if bool({'rmssd','nrmssd'} & set(flist)):
        rmssd = np.sqrt(np.mean(sd**2, axis=1)) 

    #calculate hr
    if bool({'meanhr','maxhr','minhr','medianhr','sdhr'} & set(flist)):
        hr = 60000/rri


    if 'meanrr' in flist:
        ix = flist.index('meanrr')
        features_arr[:,ix] = meanrr
    
    if 'sdrr' in flist:
        ix = flist.index('sdrr')
        features_arr[:,ix] = sdrr
    
    if 'medianrr' in flist:
        ix = flist.index('medianrr')
        features_arr[:,ix] = np.median(rri,axis=1)
    
    if 'rangerr' in flist:
        ix = flist.index('rangerr')
        features_arr[:,ix] = np.amax(rri,axis=1)-np.amin(rri,axis=1)

    if 'nsdrr' in flist:
        ix = flist.index('nsdrr')
        features_arr[:,ix] = sdrr/meanrr

    if 'sdsd' in flist:
        ix = flist.index('sdsd')
        features_arr[:,ix] = np.std(sd,axis=1)

    #Root mean square of successive RR interval differences
    if 'rmssd' in flist:
        ix = flist.index('rmssd')
        features_arr[:,ix] = rmssd 

    if 'nrmssd' in flist:
        ix = flist.index('nrmssd')
        features_arr[:,ix] = rmssd/meanrr

    if 'prr50' in flist:
        ix = flist.index('prr50')
        prr50 = 100 * np.sum(np.abs(sd)>5, axis=1)/sd.shape[1]
        features_arr[:,ix] = prr50

    if 'meanhr' in flist:
        ix = flist.index('meanhr')
        features_arr[:,ix] = np.mean(hr,axis=1)

    if 'maxhr' in flist:
        ix = flist.index('maxhr')
        features_arr[:,ix] = np.amax(hr,axis=1)
    
    if 'minhr' in flist:
        ix = flist.index('minhr')
        features_arr[:,ix] = np.amin(hr,axis=1)
   
    if 'medianhr' in flist:
        ix = flist.index('medianhr')
        features_arr[:,ix] = np.median(hr,axis=1)

    if 'sdhr' in flist:
        ix = flist.index('sdhr')
        features_arr[:,ix] = np.std(hr,axis=1)

    return features_arr 