from __future__ import print_function

import scipy.io

from Unet_scale2 import unet

from data_loader import DataLoader
from config import config_func

use_interp = True
# epoch = '69'

size_HR = 128
size_LR = 64
NA = 0.2        # NA of 대물렌즈
LEDgap = 0.5   # gap between adjacent LEDs
LEDheight = 10  # distance bewteen the LED matrix and the sample
arraysize = 9

# load_dir = 'HR%d_LR%d_NA%.2f_gap%.2f_height%.1f_arraysize_%d' % (size_HR, size_LR, NA, LEDgap, LEDheight, arraysize)
# if use_interp == True:
#     load_dir2 = load_dir + '_interp'  
# else:
#     load_dir2 = load_dir
    
model = unet()


opt = config_func()


data_loader_test = DataLoader(opt.dataset_name, 'Test')
    
Settings = {
    "size_HR"    : 128 ,
    "size_LR"    : 32  ,
   
    "arraysize"  : 9   ,
    #parameter for Imaging System
    "NA"         : 0.5     , # NA of 대물렌즈
    "LEDgap"     : 5    , # gap between adjacent LEDs
    "LEDheight"  :  10   , # distance bewteen the LED matrix and the sample

}

test_data = data_loader_test.load_data(opt, Settings)
test_FPM_datasets = test_data['FPM_datasets']

model.load_weights('./SavedModels/32_to_128_epoch%d.hdf5')

Test_pred_Base_32 = model.predict(test_FPM_datasets , batch_size=1)

scipy.io.savemat('Pred/Test_pred_Base_32.mat', mdict={'Test_pred_Base_32': Test_pred_Base_32})
 
 

# model.summary()

# if use_interp == True:    
#     X_data = np.load('../../Data/%s/LR_Amp_Test_interp.npy' % load_dir)
# else:
#     X_data = np.load('../../Data/%s/LR_Amp_Test.npy' % load_dir)


