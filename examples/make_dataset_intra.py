"""
This example shows how to create the datasets intra-patient.
"""
import sys
sys.path.insert(0, '.')
from pyecg.utils import *
from pyecg.data_handling import DataHandling
from pyecg.data_augmentation import *
from pyecg.data_handling import DS1,DS2 

# Intra-patient
dh = DataHandling(base_path='./data', win=[120,180],remove_bl=False,lowpass=False)
dh.save_dataset_intra(records=DS1, split_ratio=0.7, save_file_prefix='intra') 

# Loading the sets
train_ds = dh.load_data(file_name='intra_train.beat')
test_ds = dh.load_data(file_name='intra_test.beat')

# Number of samples per class
stat_report = dh.report_stats_table([train_ds['labels'],
                                    test_ds['labels']], 
                                    ['Train','Test']) 