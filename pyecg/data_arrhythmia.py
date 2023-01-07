

import numpy as np
import math
import time
import tensorflow as tf
from tqdm import tqdm
from pyecg.data_info import *
from pyecg.data_handling import DataHandling
from pyecg.utils import save_data
from pyecg.data_preprocessing import denoise_signal
from pyecg.features import get_hrv_features 



def get_ecg_record(record_num=106):
	"""
	Returns:
		Signal and its annotations as a dictionary with keys: 
			signal, r_locations, r_labels, rhythms, rhythms_locations
	"""

	dh = DataHandling(base_path='../data')
	rec_dict = dh.get_signal_data(record_num=record_num, return_dict=True)
	rec_dict['signal'] = denoise_signal(rec_dict['signal'],remove_bl=True,lowpass=False)
	return rec_dict 

def full_annotate_arr(record):
	"""Fully annotate a single recorded signal.

    Args:
    	record: 
    		A dictionary with keys: signal,r_locations,
    		r_labels,rhythms,rhythms_locations

    Returns:
		A 2d list-->[signal, full_ann]. First dim is the original signal. 
		Second dim is a list that has the same size as the input signal with 
		elements as the arrhythmia class at each index 
		like: ['(N','(N','(N','(N','AFIB','AFIB','AFIB',...]

    """

	signal,_,_,rhythms,rhythms_locations = record.values()
	sig_length = len(signal)
	full_ann = []
	full_ann = ['unlab']*len(signal)
	for i in range(len(rhythms_locations)):
		remained = sig_length-rhythms_locations[i]
		full_ann[rhythms_locations[i]:]=[rhythms[i]]*remained
	record_full = [signal, full_ann]
	return record_full 

def get_all_annotated_records(rec_list):
	"""
	Args:
		rec_list:
			List of records.
	Returns:
	 	A list containing a dict for each record. [rec1,rec2,....].
	 	Each rec is a dictionary with keys: 
	 				signal, full_ann, r_locations, r_labels,rhythms,rhythms_locations.
	"""

	all_recs = []
	for rec_no in tqdm(rec_list):
		rec_dict = get_ecg_record(record_num=rec_no)
		rec_dict['full_ann'] = full_annotate_arr(rec_dict)[1]
		all_recs.append(rec_dict)
	return all_recs 

def make_samples_info(annotated_records, win_size=30*360, stride=36):
	"""
	Args:
		A list containing a dict for each record. [rec1,rec2,....]. Each rec is a dictionary.
	Returns:
	returns a 2d list. Each inner list: [index,record_no,start_win,end_win,label]
	[[record_no,start_win,end_win,label],[record_no,start_win,end_win,label], ...]
	eg: [[10,500,800,'AFIB'],[],...]
	"""

	stride = int(stride)
	win_size = int(win_size)

	samples_info = []

	for rec_no in tqdm(range(len(annotated_records))):
		signal = annotated_records[rec_no]['signal']
		full_ann = annotated_records[rec_no]['full_ann']
		assert len(signal)==len(full_ann), 'signal and annotation must have the same length!'

		end=win_size
		while end<len(full_ann):
			start=int(end-win_size)
			#unique arrhythmia type in each segment
			if len(set(full_ann[start:end])) == 1:
				label = full_ann[start]
				samples_info.append([rec_no,start,end,label])
			end += stride
		time.sleep(3)
	return samples_info 

def save_samples_arr(rec_list=DS1,file_path=None,stride=36):
	annotated_records = get_all_annotated_records(rec_list)
	samples_info = make_samples_info(annotated_records,stride=stride)
	data = [annotated_records, samples_info]
	save_data(data, file_path=file_path)
	return data 




class ECGSequence(tf.keras.utils.Sequence):
	"""data is a 2d list.
			 [[signal1, full_ann1],[signal2, full_ann2],...]
			only the signal parts are used.
	   samples_info is a 2d list. 
			[[index,record_no,start_win,end_win,label],[index,record_no,start_win,end_win,label], ...]
			eg: [[1,10,500,800,'AFIB'],[],...]
	"""

	def __init__(self, data, samples_info, class_labels=None, 
					batch_size=128, shuffle=True, denoise=True):
		"""
		Args:
			data: A list containing a dict for each record. [rec1,rec2,....].
	 			  Each rec is a dictionary with keys: 
	 			  signal, full_ann, r_locations, r_labels,rhythms,rhythms_locations.
		"""
		self.shuffle = shuffle
		self.denoise = denoise
		self.batch_size = batch_size
		self.data = data
		self.samples_info = samples_info
		self.class_labels = class_labels
		self.on_epoch_end()

	def __len__(self):
		return math.ceil(len(self.samples_info) / self.batch_size)

	def __getitem__(self, idx):
		batch_samples = self.samples_info[idx * self.batch_size:(idx + 1) * self.batch_size]

		batch_seq = []
		batch_label = []
		batch_rri = []
		for sample in batch_samples:
			#eg sample:[10,500,800,'AFIB'] ::: [rec,start,end,label]
			rec_no = sample[0]
			start = sample[1]
			end = sample[2]
			label = sample[3]
			if self.class_labels != None:
				label = self.get_integer(label)

			seq = self.data[rec_no]['signal'][start:end]

			batch_seq.append(seq)
			batch_label.append(label)

			rri = self.get_rri(rec_no,start,end)
			batch_rri.append(rri)

		batch_rri_feat = self.get_rri_features(np.array(batch_rri)*1000)

		#return np.array(batch_seq),np.array(batch_label)
		return [np.array(batch_seq), np.array(batch_rri), batch_rri_feat], np.array(batch_label)

	def on_epoch_end(self):
		#after each epoch shuffles the samples
		if self.shuffle:
			np.random.shuffle(self.samples_info)

	def get_integer(self,label):
		#text label to integer
		return self.class_labels.index(label)

	def get_rri(self,rec_no,start,end):
		r_locations = np.asarray(self.data[rec_no]['r_locations']) #entire record
		inds = np.where((r_locations>=start) & (r_locations<end))
		rpeak_locs = list(r_locations[inds])
		rri = [(rpeak_locs[i+1]-rpeak_locs[i])/360.0 for i in range(0,len(rpeak_locs)-1)]
		#padding for 30sec---len=150
		#print(rri)
		rri_zeropadded = np.zeros(150)
		rri_zeropadded[:len(rri)] = rri
		#print(rri_zeropadded)
		rri_zeropadded = rri_zeropadded.tolist()

		rri_zeropadded = rri_zeropadded[:20] #TODO

		return rri_zeropadded

	def get_rri_features(self,arr):
		#features = ['max','min']
		return get_hrv_features(arr)











