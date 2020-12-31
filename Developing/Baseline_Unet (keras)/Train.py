from __future__ import print_function, division

from importlib import import_module
import os

import numpy as np

from keras.optimizers import Adam
from keras.models import load_model

from config import config_func
import time
import math
import matplotlib.pyplot as plt

from data_loader import DataLoader

use_interp = False
    
Settings = {
    "size_HR"    : 128,
    "size_LR"    : 32,
   
    "arraysize"  : 5,    # parameter for Imaging System
    "NA"         : 0.15, # NA of 대물렌즈
    "LEDgap"     : 12,   # gap between adjacent LEDs
    "LEDheight"  : 50,   # distance bewteen the LED matrix and the sample
    "spsize"     : 1.5e-6# sampling pixel size of the camera
}

opt = config_func()

if use_interp == True:
    model_name = 'unet_scale1'
else:
    model_name = 'unet_scale%d' % (Settings["size_HR"] // Settings["size_LR"])        

model = getattr(import_module('Model'), model_name)(sx=Settings["size_LR"], sy=Settings["size_LR"], nch=Settings["arraysize"] ** 2)

add_path_str = '%d_to_%s_arraysize%s' % (Settings["size_LR"], Settings["size_HR"], Settings["arraysize"])

os.makedirs('SavedModels_%s' % add_path_str, exist_ok=True)
os.makedirs('OutFigures_%s' % add_path_str,  exist_ok=True)

if opt.epoch != 0:
    model = load_model('./SavedModels_%s/epoch%d.hdf5' % (add_path_str, opt.epoch))

optimizer = Adam(lr=opt.lr, beta_1=0.5, beta_2=0.999, clipvalue = opt.clip_value)

data_loader_train = DataLoader(opt.dataset_name, 'Train')
data_loader_valid = DataLoader(opt.dataset_name, 'Valid')

train_num = data_loader_train.len_data()
batch_num = train_num // opt.batch_size

start_time = time.time()

valid_data = data_loader_valid.load_data(opt , Settings)
valid_FPM_datasets = valid_data['FPM_datasets']
valid_amp = np.squeeze(valid_data['amp'])
valid_phase =  np.squeeze(valid_data['phase'])

model.compile(loss=['mse'],  optimizer=optimizer)

for epoch in range(opt.epoch, opt.n_epochs):
    for batch in range(batch_num):
        
        b_start_time = time.time()
        # Train the generator
        train_data = data_loader_train.load_data(opt , Settings)
        
        train_loss = model.train_on_batch(train_data['FPM_datasets'], np.concatenate([train_data['amp'], train_data['phase']], axis=-1))

        if batch%opt.checkpoint_interval==0:
        # validate during training
            valid_pred = model.predict(valid_FPM_datasets , batch_size=1)
            
            
            valid_pred_amp   = valid_pred[:,:,:,0]
            valid_pred_phase = valid_pred[:,:,:,1]
            

            valid_amp_loss = math.sqrt(((valid_amp - valid_pred_amp) ** 2).mean())
            valid_phase_loss = math.sqrt(((valid_phase - valid_pred_phase) ** 2).mean())  
            
            valid_loss = valid_amp_loss + valid_phase_loss
            
            e_time = time.time() - start_time
            bat_time = time.time() - b_start_time
            # Plot the progress
            print("[epoch:%d/%d] [batch:%d/%d] [train_loss:%.4f] [valid_loss(RMSE):%.4f (Amp) + %.4f (Phase)= %.4f][elapsed_time:%d:%2d:%2d] [Batch_time : %d:%d]" %
                   (epoch, opt.n_epochs, batch, batch_num, train_loss, valid_amp_loss, valid_phase_loss,valid_loss, e_time // 3600, e_time % 3600 // 60, e_time % 60 , bat_time %3600//60 , bat_time %60))
            
            batches_done = epoch * batch_num + batch
            samples_done = batches_done * opt.batch_size
    
          
            # Save validation results in every sample_interval
            if batch % opt.sample_interval == 0:
            # if batches_done % opt.sample_interval == 0:
                amp_recon = np.zeros((1024,1024))
                phase_recon = np.zeros((1024,1024))
                          
                amp_valid_compare = np.zeros((1024,1024))
                phase_valid_compare = np.zeros((1024,1024))
                
                zzz = 0
                for nn in range(8):
                    for nnn in range(8):
                        amp_recon[  (nn*128) : (nn+1)*128, (nnn*128) : (nnn+1)*128 ] = valid_pred_amp[64 + zzz,:,:]
                        phase_recon[(nn*128) : (nn+1)*128, (nnn*128) : (nnn+1)*128 ] = valid_pred_phase[64+zzz,:,:]
                        amp_valid_compare[  (nn*128) : (nn+1)*128, (nnn*128) : (nnn+1)*128 ] = valid_amp[64+zzz,:,:]
                        phase_valid_compare[  (nn*128) : (nn+1)*128, (nnn*128) : (nnn+1)*128 ] = valid_phase[64+zzz,:,:]
                        
                        zzz += 1
                
                r, c = 2, 2
                titles = ['Reconstructed_amp (epoch=%d, RMSE=%.4f)' % (epoch, valid_amp_loss),
                          'Reconstructed_phase (epoch=%d, RMSE=%.4f)' % (epoch, valid_phase_loss),
                          'Original_amp' , 'Original_phase']
                
    
                fig, axs = plt.subplots(r, c)
                
                axs[0, 0].imshow(amp_recon, cmap='gray', vmin=0, vmax=1)
                axs[0, 0].set_title(titles[0],fontdict={'fontsize': 4})
                axs[0, 0].axis('off')
                
                axs[0, 1].imshow(phase_recon, cmap='gray', vmin=0, vmax=1)
                axs[0, 1].set_title(titles[1],fontdict={'fontsize': 4})
                axs[0, 1].axis('off')
                
                axs[1, 0].imshow(amp_valid_compare, cmap='gray', vmin=0, vmax=1)
                axs[1, 0].set_title(titles[2],fontdict={'fontsize': 4})
                axs[1, 0].axis('off')
                
                axs[1, 1].imshow(phase_valid_compare, cmap='gray', vmin=0, vmax=1)
                axs[1, 1].set_title(titles[3],fontdict={'fontsize': 4})
                axs[1, 1].axis('off')
                      
                fig.savefig('./OutFigures_%s/epoch%d_%d_RMSE%.4f.png' %
                    (add_path_str, epoch, samples_done, valid_loss), dpi = 150)
                plt.close()
                
            del valid_pred ,valid_pred_amp, valid_pred_phase
        del train_loss , train_data
        
    # Save model checkpoints      
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # model.save('SavedModels/%s_to_%s_epoch%d.hdf5' % (Settings["size_LR"] , Settings["size_HR"], epoch + 1))
        model.save('SavedModels_%s/epoch%d.hdf5' % (add_path_str, epoch + 1))