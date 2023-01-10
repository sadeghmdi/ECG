"""
This example shows how to create the datasets inter-patient.
"""
from pyecg.data_handling import DS1, DS2
from pyecg.data_augmentation import *
from pyecg.data_handling import DataHandling
from pyecg.utils import *
import sys
sys.path.insert(0, '.')

# Inter-patient
# Lets create a DataHandling object.the train set by creating
dh = DataHandling(base_path='./data',
                  win=[500, 500], remove_bl=False, lowpass=False)

# use the save_dataset method to create the dataset file.
# The file will be saved in the base data directory.
dh.save_dataset(records=DS1[:18], save_file_name='train.beat')

# In a similar way for validation and test sets
dh.save_dataset(records=DS1[18:], save_file_name='val.beat')
dh.save_dataset(records=DS2, save_file_name='test.beat')

# Loading the sets
train_ds = dh.load_data(file_name='train.beat')
val_ds = dh.load_data(file_name='val.beat')
test_ds = dh.load_data(file_name='test.beat')

# Number of samples per class
stat_report = dh.report_stats_table([train_ds['labels'],
                                    val_ds['labels'],
                                    test_ds['labels']],
                                    ['Train', 'Val', 'Test'])
print(stat_report)
