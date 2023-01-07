


import numpy as np
import math
import time
import tensorflow as tf
from tqdm import tqdm
from pyecg.data_info import *
from pyecg.data_handling import DataHandling
from pyecg.utils import save_data
from pyecg.data_preprocessing import denoise_signal
from scipy import stats



def get_ecg_record(record_num=106):
	'''
		return the signal and annotations as a dictionary
		keys: signal,r_locations,r_labels,rhythms,rhythms_locations
	'''
	dh = DataHandling(base_path='../data')
	rec_dict = dh.get_signal_data(record_num=record_num, return_dict=True)
	rec_dict['signal'] = denoise_signal(rec_dict['signal'],remove_bl=True,lowpass=False)
	return rec_dict

def full_annotate_rpeak(record):
	'''fully annotate a single recorded signal. 
	Args:
		record = a dictionary.
			keys: signal,r_locations,r_labels,rhythms,rhythms_locations
	Returns:
		a list of zeros. At any index with an rpeak the zero is changed to label.
		[00000N00000000000000L00000...]
	'''
	signal,r_locations,r_labels,_,_ = record.values()

	full_seq = [0]*len(signal)
	for i,loc in enumerate(r_locations):
		full_seq[loc] = r_labels[i] 

	return [signal,full_seq]

def get_all_annotated_records(rec_list):
	'''
	Returns:
		 a list containing [signal, full_ann] for all records in rec_list
		[[signal1, full_ann1],[signal2, full_ann2],...]
	'''
	all_recs = [] 
	for rec_no in tqdm(rec_list):
		rec_dict = get_ecg_record(record_num=rec_no)
		d = full_annotate_rpeak(rec_dict)
		all_recs.append(d)
	return all_recs

def make_samples_info(annotated_records, win_size=30*360, stride=256, interval=128, binary=True):
	'''
	Args:
		annotated_records: [[signal1, full_ann1],[signal2, full_ann2],...]
		win_size : the length of each extracted sample
		stride : the stride for extracted samples.
		interval : the output interval for labels.
		binary : if True 1 is replaced instead of labels
	Returns:
			a 2d list. Each inner list: [record_no,start_win,end_win,label]
			[[record_no,start_win,end_win,label],[record_no,start_win,end_win,label], ...]
			label is a list. Elements are rpeak labels for each interval.
			eg: [[10,500,800, [000000N00000000L00] ],[],...]
	'''
	stride = int(stride)
	win_size = int(win_size)
	interval = int(interval)

	samples_info = []
	
	#each record
	for rec_no in tqdm(range(len(annotated_records))):
		signal, full_ann = annotated_records[rec_no]
		assert len(signal)==len(full_ann), 'signal and annotation must have the same length!'

		#each extracted segment
		end=win_size
		while end<len(full_ann):
			start=int(end-win_size)
			seg = full_ann[start:end]
			labels_seq = []

			#each subsegment
			for i in range(0,len(seg),interval): 
				subseg = seg[i:i+interval]
				if any(subseg):
					nonzero = [l for l in subseg if l!=0]
					lb = nonzero[0]
					if binary:
						lb=1
					labels_seq.append(lb)
				else:
					lb = 0
					labels_seq.append(lb)

			samples_info.append([rec_no,start,end,labels_seq])
			end += stride
		time.sleep(3)

	return samples_info


def save_samples_rpeak(rec_list,file_path, win_size, stride, interval, binary):
	annotated_records = get_all_annotated_records(rec_list)
	samples_info = make_samples_info(annotated_records, win_size, stride, interval, binary) 
	data = [annotated_records,samples_info]
	save_data(data, file_path=file_path)
	return data




class ECGSequence(tf.keras.utils.Sequence):
	'''
	Args:
		data: is a 2d list.
			 [[signal1, full_ann1],[signal2, full_ann2],...]
			only the signal parts are used.
		samples_info: is a 2d list. 
			[[record_no,start_win,end_win,label],[record_no,start_win,end_win,[labels] ], ...]
			eg: [[10,500,800,[0,0,0,'N',0,0...],[],...]

	Returns: 
		batch_x: 2d array of (batch,segments,features) 	  (batch_size, 300, 2) 
		batch_y: 2d array of (batch,label_list) 			(batch_size, 300)

	'''

	def __init__(
		self, 
		data, 
		samples_info, 
		batch_size,
		binary=True,
		raw=False, 
		class_labels=None,
		shuffle=True
		):

		self.shuffle = shuffle
		self.binary = binary
		self.raw=raw
		self.batch_size = batch_size
		self.data = data
		self.samples_info = samples_info
		self.class_labels = class_labels
		self.on_epoch_end()

	def __len__(self):
		return math.ceil(len(self.samples_info) / self.batch_size)

	def __getitem__(self, idx):
		batch_samples = self.samples_info[idx * self.batch_size:(idx + 1) * self.batch_size]

		batch_x = []
		batch_y = []

		for sample in batch_samples:
			#eg sample:[10,500,800,[0,0,0,'N',0,0...] >>[rec,start,end,label]
			rec_no = sample[0]
			start = sample[1]
			end = sample[2]
			label = sample[3]

			seq = self.data[rec_no][0][start:end]
			#processing steps on the signal fraggment
			if self.raw == False:
				seq = self.proc_steps(seq) 
				batch_x.append(list(seq))
			else:
				batch_x.append(seq)


			if self.binary:
				label = self.get_binary(label)
			if self.class_labels !=None :
				label = self.get_integer(label)
			batch_y.append(label)

		return np.array(batch_x), np.array(batch_y)

	def on_epoch_end(self):
		#after each epoch shuffles the samples
		if self.shuffle:
			np.random.shuffle(self.samples_info)

	def get_binary(self,label):
		#text label to integer 0,1 label
		label = [1 if item!=0 else 0 for item in label]
		return label

	def get_integer(self,label):
		#text label to integer >> 0,1,2,...
		label = [list(self.class_labels).index(str(item)) for item in label]
		return label

	def proc_steps(self,seq):
		#get a 1d seq and return a multidim list.
		#each feature is an aggragate of a subseq.
		#stride =36
		stride =72
		b = int(len(seq)/stride)
		subseqs = []
		for i in range(b):
			subseq = seq[i*stride:(i+1)*stride]
			subseqs.append(subseq)  
		subseqs = np.array(subseqs)  #36---> (300,36)

		f1 = np.amax(subseqs,axis=1)
		f2 = np.amin(subseqs,axis=1)
		f3 = np.mean(subseqs,axis=1)
		f4 = np.std(subseqs,axis=1)
		f5 = np.median(subseqs,axis=1)
		f6 = stats.skew(subseqs,axis=1)
		f7 = stats.kurtosis(subseqs,axis=1)

		sqr = subseqs**2
		f8 = np.amax(sqr,axis=1)
		f9 = np.amin(sqr,axis=1)
		f10 = np.mean(sqr,axis=1)
		f11 = np.std(sqr,axis=1)
		f12 = np.median(sqr,axis=1)
		f13 = stats.skew(sqr,axis=1)
		f14 = stats.kurtosis(sqr,axis=1)

		feats = np.vstack((f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14))

		return feats.T





