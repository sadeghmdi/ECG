


import sys
sys.path.insert(0, '.')
from pyecg.utils import *
from pyecg.data_handling import DataHandling
from pyecg.data_augmentation import *
from pyecg.data_handling import DS1,DS2 



#train set
dh = DataHandling(base_path='data', win=[400,400],remove_bl=True,lowpass=True)
dh.save_dataset(records=DS1[19:], save_file_name='train.beat')
ds = dh.load_data(file_name='train.beat')
x_train, r_train, y_train = ds.values()


#import time
#time.sleep(10)
#val set
#dh.save_dataset(records=DS1[:11], save_file_name='val.beat')
#ds = dh.load_data(file_name='val.beat')
#x_val, r_val, y_val = ds.values()

#x_train.shape, r_train.shape, y_train.shape, x_val.shape
