import numpy as np
import pandas as pd
import wfdb
import pickle
import os
import copy
from tqdm import tqdm
from pyecg.data_info import *
from pyecg.data_preprocessing import *
from pyecg.beat_info import BeatInfo




class DataHandling:
    """
    Method save_dataset creates a file containing signal fragments
                and their corresponding label

    example:
            dh = DataHandling(base_path='../data')
            dh.save_dataset(records=DS2,save_file_name='DS2.dat')
            x_test, y_test = dh.load_data(load_file_name='DS2.dat')
            statReport = dh.report_stats_table([y_test],['test'])
    """
    def __init__(self, base_path=os.getcwd(), data_path=DATA_PATH, win=[60,120],
                                        remove_bl=True,lowpass=True,fs=360,cutoff=45,order=15):

        self.base_path = base_path
        self.data_path = os.path.join(self.base_path, data_path)
        self.syms = [k for k,v in MAP_AAMI.items()]
        self.win = win
        self.remove_bl = remove_bl
        self.lowpass = lowpass
        self.fs = fs
        self.cutoff = cutoff
        self.order = order

    def get_signal_data(self, record_num=106, return_dict=True):
        record = wfdb.rdrecord(self.data_path + str(record_num), channel_names=['MLII'])
        annotation = wfdb.rdann(self.data_path + str(record_num), 'atr')
        signal = record.p_signal[2:,1]
        ann_locations = annotation.sample
        symbol = annotation.symbol
        aux = annotation.aux_note
        aux=[txt.rstrip('\x0') for txt in aux]
        signal = denoise_signal(signal,remove_bl=self.remove_bl,
                                lowpass=self.lowpass,fs=self.fs,
                                cutoff=self.cutoff,order=self.order) 
        r_labels = []
        r_locations = []
        rhythms = []
        rhythms_locations = []
        for i in range(len(ann_locations)):
            if symbol[i] in self.syms:
                r_labels.append(symbol[i])
                r_locations.append(ann_locations[i])

            if aux[i] in RHYTHM_TYPES:
                rhythms.append(aux[i])
                rhythms_locations.append(ann_locations[i])

        if return_dict == True:
            return {'signal':signal, 
                    'r_locations':r_locations, 'r_labels':r_labels, 
                    'rhythms':rhythms, 'rhythms_locations':rhythms_locations}
        else:
            annots = [None]*len(signal)
            for i in range(len(an_locations)):
                annots[ann_location[i]] = symbols[j]
                if aux[i] !='':
                    annots[ann_locations[i]] = aux[i]
            sig_dict = {'time':range(len(signal)), 'signal':signal, 'annots':annots} 
            return pd.DataFrame(sig_dict)  

    def make_frags(self, signal, r_locations=None, r_label=None, num_pre_rr=10, num_post_rr=10):
        win = self.win
        frags = []
        beat_types = []
        r_locs = []
        s_idxs = [] #start indexes
        for i in range(num_pre_r,(len(r_locations)-num_post_rr)):
            start_idx = r_locations[i]-win[1]
            end_idx = r_locations[i]+win[0]
            seg = signal[start_idx:end_idx]
            if len(seg) == win[0]+win[1]:
                frags.append(seg)
                beat_types.append(r_label[j])
                r_locs.append(list(r_locations[i-num_pre_rr : i+num_post_rr+1]))
                s_idx.append(start_idx)
        signal_frags = np.array(frags)
        return signal_frags, beat_types, r_locs, s_idxs 

    def make_dataset(self, records=None):
        """ Creates the full dataset 
        xds : numpy array of the signal fragments. shape=(#fragments, length_of_fragment)
        yds : list of corresponding types.  size=#fragments
        """
        if records is None:
            records = RECORDS
        xds=[]
        yds=[]
        rds=[]
        start_idxs=[]
        rec_num_list=[] #debug
        for i in tqdm(range(len(records))):
            rec_dict = self.get_signal_data(record_num=records[i+1])
            signal,r_locations,r_labels,_,_ = rec_dict.values() 
            signal_frags, beat_types, r_locs, s_idxs= self.make_frags(signal, r_locations, r_labels)
            start_idxs.extend(s_idx)
            rec_num_list.extend([records[i]]*len(start_idxs))
            xds = xds + signal_frags
            yds = yds + beat_types 
            rds = rds + r_locs 
            i += 1
        xds = np.array(xds)
        yds = np.array(yds)

        beat_feats,labels = self.beat_info_feat({'waveforms':xds,
                                          'rpeaks':rds,
                                          'rec_nums': rec_num_list,
                                          'start_idxs': start_idx,
                                          'labels':yds}, 
                                           beat_loc=10
                                           )
        beat_feats = [[i for i in beat_feat.values()] for beat_feat in beat_feats]
        beat_feats = np.array(beat_feats)

        return {'waveforms':xds, 'beat_feats':beat_feat, 'labels':np.array(labels)}

    def beat_info_feat(self, data, beat_loc=10):
        #takes as input a dictionary containing an arrays of signal fragments and
        #their corrsponding r locations
        assert len(data['waveforms'])==len(data['start_idxs']), "len must be equal!"

        beatinfo = BeatInfo(beat_loc=beat_locs)
        features=[]
        labels= []
        #beats loop
        for i in tqdm(range(len(data['rpeak']))): 
            
            beatinfo({'waveform':data['waveforms'][i], 
                      'rpeak':data['rpeaks'][i], 
                      'rec_num': data['rec_nums'][i],
                      'start_idx':data['start_idxs'][i],
                      'label': data['labels'][i]
                      })
            #check if the beat correctly annotated
            if (beatinfo.bwaveform) is None:
                continue

            res = {}
            res['post_rri'] = beatinfo.post_rri()
            res['pre_rri'] = beatinfo.pre_rri() 
            res['ratio_post_pre'] = beatinfo.ratio_post_pre()
            res['diff_post_pre'] = beatinfo.diff_post_pre()
            res['diff_post_pre_nr'] = beatinfo.diff_post_pre_nr()
            res['mean_rri'] = beatinfo.mean_rri()
            res['rms_rri'] = beatinfo.rms_rri()
            res['std_rri'] = beatinfo.std_rri()
            res['ratio_pre_avg'] = beatinfo.ratio_pre_avg()
            res['ratio_post_avg'] = beatinfo.ratio_post_avg()
            res['ratio_pre_rms'] = beatinfo.ratio_pre_rms()
            res['ratio_post_rms'] = beatinfo.ratio_post_rms()
            res['diff_pre_avg_nr'] = beatinfo.diff_pre_avg_nr()
            res['diff_post_avg_nr'] = beatinfo.diff_post_avg_nr()
            res['compensate_ratio'] = beatinfo.compensate_ratio()  
            res['compensate_diff_nr'] = beatinfo.compensate_diff_nr()
            res['heart_rate'] = beatinfo.heart_rate()

            res['post_sdrri'] = beatinfo.post_sdrri()
            res['pre_sdrri'] = beatinfo.pre_sdrri() 
            res['diff_post_pre_sdrri'] = beatinfo.diff_post_pre_sdrri()
            res['diff_post_pre_nr_sdrri'] = beatinfo.diff_post_pre_nr_sdrri()
            res['mean_sdrri'] = beatinfo.mean_sdrri()
            res['absmean_sdrri'] = beatinfo.absmean_sdrri()
            res['rms_sdrri'] = beatinfo.rms_sdrri()
            res['std_sdrri'] = beatinfo.std_sdrri()
            res['diff_on_avg_sdrri_nr'] = beatinfo.diff_on_avg_sdrri_nr()
            res['ratio_pre_rms_sdrri'] = beatinfo.ratio_pre_rms_sdrri()
            res['ratio_post_rms_sdrri'] = beatinfo.ratio_post_rms_sdrri()
            res['ratio_on_rms_sdrri'] = beatinfo.ratio_on_rms_sdrri()
            res['ratio_pre_absmean_sdrri'] = beatinfo.ratio_pre_absmean_sdrri()
            res['ratio_post_absmean_sdrri'] = beatinfo.ratio_post_absmean_sdrri()
            res['ratio_on_absmean_sdrri'] = beatinfo.ratio_on_absmean_sdrri()
            #res['ratio_post_on_sdrri'] = beatinfo.ratio_post_on_sdrri()
            #res['ratio_on_pre_sdrri'] = beatinfo.ratio_on_pre_sdrri()
            #res['ratio_post_pre_sdrri'] = beatinfo.ratio_post_pre_sdrri()
            res['diff_post_on_sdrri'] = beatinfo.diff_post_on_sdrri()
            res['diff_on_pre_sdrri'] = beatinfo.diff_on_pre_sdrri()

            res['beat_max'] = beatinfo.beat_max()
            res['beat_min'] = beatinfo.beat_min()
            res['maxmin_diff'] = beatinfo.maxmin_diff()
            res['beat_mean'] = beatinfo.beat_mean()
            res['beat_std'] = beatinfo.beat_std()
            res['beat_skewness'] = beatinfo.beat_skewness()
            res['beat_kurtosis'] = beatinfo.beat_kurtosis()
            res['beat_rms'] = beatinfo.beat_rms() 
            res['pr_interval'] = beatinfo.pr_interval() 
            res['qs_interval'] = beatinfo.qs_interval() 

            #features of splitted beat waveform --10
            ten_segs,aggs = beatinfo.sub_segs(n_subsegs=10)  #many features a lis of lists
            for l in range(len(ten_segs)):
                for it in range(len(ten_segs[l])):
                    key = 'mt'+str(l+1)+str('_seg')+str(it+1)
                    res[key] =  ten_segs[l][it]

            for i in range(len(aggs)):
                for j in range(len(aggs[i])):
                    key = 'agg'+str(i+1)+'mt'+str(j+1)
                    res[key] =  aggs[i][j]  

            #features of splitted beat waveform -- 3 pqrst
            pqrst_segs,aggs = beatinfo.pqrst_segs()  #many features a lis of lists
            for l in range(len(list(pqrst_segs))):
                for it in range(len(pqrst_segs[l])):
                    key = 'mt'+str(l+1)+str('_pqrst')+str(it+1)
                    res[key] =  pqrst_segs[l][it]

            for i in range(len(list(aggs))):
                for j in range(len(aggs[i])):
                    key = 'aggpqrst'+str(i+1)+'mt'+str(j+1)
                    res[key] =  aggs[i][j]



            #fft features
            yf = beatinfo.fft_features()
            for i in range(len(yf)):
                key = 'fft_'+str(i+1)
                #res[key] = yf[i]

            #append for each beat
            features.append(res)
            #only append labels for beats that are correct
            labels.append(beatinfo.label) 

        return features,labels

    def clean_irrelevant_data(self, ds):
        """ Removes irrelevant symbols"""
        yds = ds['labels']
        xds = ds['waveforms']
        rds = ds['beat_feats']
        indexes_rm = [i for i,item in enumerate(yds) if item not in self.syms]
        #indexes_rm = np.where(np.invert(np.isin(yds, self.syms)))[0]
        xds = np.delete(xds, indexes_rm, axis=0)
        rds = np.delete(rds, indexes_rm, axis=0)
        yds = np.delete(yds, indexes_rm, axis=0)
        #ydsc = [it for ind,it in enumerate(yds) if ind not in indexes_rm]
        return {'waveforms':xds,'beat_feats':rds,'labels':yds}

    def search_label(self, inp, sym='N'):
        """return the corresponding indexes for a patricular label"""
        if isinstance(inp, dict):
            yds = list(inp['labels'])
        elif isinstance(inp, np.ndarray):
            yds = inp
        else:
            raise TypeError("input must be a dictionary or a numpy array!")
        indexes = [i for i,item in enumerate(yds) if item==sym]
        return indexes

    def report_stats(self, yds_list):
        """Number of samples of each class in the array"""
        res_list = []
        for yds in yds_list:
            res = {}
            for sym in self.syms:
                indexes = [i for i,item in enumerate(yds) if item==sym]
                res[sym]=len(indexes)
            res_list.append(res)
        return res_list

    def report_stats_table(self, yds_list, name_list=[]):
        """Number of samples of each class in the array shown in tabular form"""
        if len(name_list) == len(yds_list):
            indx =  name_list
        else:
            indx = None
        res_list = self.report_stats(yds_list)
        df = pd.DataFrame(res_list, index=indx)
        return df

    def save_data(self, ds, save_file_name=None):
        if save_file_name is None:
            raise ValueError('Save file path is not provided!')
        save_file_path = os.path.join(self.base_path, save_file_name)       
        with open(save_file_path, 'wb') as f:
            pickle.dump(ds, f)
        print('file saved: ' + str(save_file_path))

    def save_dataset(self, records=None, clean=True, save_file_name=None):
        """Saves the signal fragments and their labels into a file"""
        if save_file_name is None:
            raise ValueError('Save file path is not provided!')
        ds = self.make_dataset(records=records)
        if clean == True:
            ds = self.clean_irrelevant_data(ds)
        self.save_data(ds, save_file_name=save_file_name)

    def load_data(self, file_name=None):
        if file_name is None:
            raise ValueError('Load file path is not provided!')
        load_file_path = os.path.join(self.base_path, file_name)    
        with open(load_file_path, 'rb') as f:
            ds = pickle.load(f)
            print('file loaded: ' + load_file_path)
            for k,v in ds.items():
                print('shape of "{}" is {}'.format(k,v.shape))
            print(self.report_stats_table([ds['labels']],[file_name]))
        return ds

    def per_record_stats(self, rec_num_list=DS1, cols=None):
        #return a table containing the number of each type in each record
        if cols==None:
            cols = self.syms
        ld = []
        for rec_num in rec_num_list:
                rec_dict = self.get_signal_data(record_num=records[i]) 
                res = np.unique(rec_dict['r_labels'], return_counts=True)
                ld.append(dict(zip(res[0],res[1])))

        df=pd.DataFrame(ld,index=rec_num_list)
        df.fillna(0,inplace=True)
        df = df.astype('int32')
        df = df[list(set(cols) & set(df.columns))]
        return df



##---------------------------
def stndr(arr,mean,std):
        X = arr.copy()
        X = (X - np.mean(X)) / np.std(X)
        return X
        
def slice_data(ds, labels_list):
    #only keep the lables in lable_list
    #ds = copy.copy(ds)
    sliced_x = ds['waveforms']
    sliced_r = ds['beat_feats'] 
    sliced_y = ds['labels'] 
    indexes_keep = []
    for sym in labels_list:
        inds = [i for i,item in enumerate(sliced_y) if item==sym]
        indexes_keep.extend(inds)
    sliced_x = sliced_x[indexes_keep]
    sliced_r = sliced_r[indexes_keep]
    sliced_y = sliced_y[indexes_keep]
    #sliced_y = [sliced_y[i] for i in indexes_keep]

    return {'waveforms':sliced_x, 'beat_feats':sliced_r, 'labels':sliced_y}

def binarize_lables(y,positive_lable,pos=1,neg=-1):
    #y a list of lables
    #positive_lable: positive class lable 
    new_y = [pos if item==positive_lable else neg for item in y]
    return new_y  

