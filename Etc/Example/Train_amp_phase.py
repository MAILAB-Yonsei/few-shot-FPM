from __future__ import print_function, division
from importlib import import_module

import numpy as np
import os

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from Unet_scale2 import unet
# from Unet_scale1 import unet

from config import config_func
import time
import math
import matplotlib.pyplot as plt

from data_loader import DataLoader



check_data = False # Generate only first data, and view the 
use_interp = False


num_train  = 335
num_val  = 15 
num_test = 50  
    
    
Settings = {
    "size_HR"    : 128 ,
    "size_LR"    : 32  ,
   
    "arraysize"  : 9   ,
    #parameter for Imaging System
    "NA"         : 0.5     , # NA of 대물렌즈
    "LEDgap"     : 5    , # gap between adjacent LEDs
    "LEDheight"  :  10   , # distance bewteen the LED matrix and the sample

}



if check_data == True:
    num_data = 1
else:
    num_data = num_train + num_val + num_test


opt = config_func()


model = unet()

model = load_model('./SavedModels/32_to_128_epoch19_val_loss_0.3635.hdf5')
optimizer = Adam(lr=opt.lr, beta_1=0.5, beta_2=0.999, clipvalue = opt.clip_value)




# chh = DataLoader(opt.dataset_name, 'Check')
# check = chh.load_data(opt , Settings)



data_loader_train = DataLoader(opt.dataset_name, 'Train')
data_loader_valid = DataLoader(opt.dataset_name, 'Valid')



train_num = data_loader_train.len_data()
batch_num = train_num // opt.batch_size


start_time = time.time()

pid = 1

valid_data = data_loader_valid.load_data(opt , Settings)
valid_FPM_datasets = valid_data['FPM_datasets']
valid_amp = np.squeeze(valid_data['amp'])
valid_phase =  np.squeeze(valid_data['phase'])


batch = 0
opt.epoch=19

model.compile(loss=['mse'],  optimizer=optimizer)

for epoch in range(opt.epoch, opt.n_epochs):
 
    for batch in range(batch_num):
        
        b_start_time = time.time()
        # Train the generator
        train_data = data_loader_train.load_data(opt , Settings)
        
        
            
        train_loss = model.train_on_batch(train_data['FPM_datasets'] , np.concatenate([train_data['amp'] , train_data['phase']], axis=-1))
        
    
        if batch%15==0:
        # validate during training
            
            
            
            valid_pred = model.predict(valid_FPM_datasets , batch_size=1)
            
            
            valid_pred_amp = valid_pred[:,:,:,0]
            valid_pred_phase = valid_pred[:,:,:,1]
            

            valid_amp_loss = math.sqrt(((valid_amp - valid_pred_amp) ** 2).mean())
            valid_phase_loss = math.sqrt(((valid_phase - valid_pred_phase) ** 2).mean())  
            
            valid_loss = valid_amp_loss + valid_phase_loss
            
            e_time = time.time() - start_time
            bat_time = time.time() - b_start_time
            # Plot the progress
            print("[epoch:%d/%d] [batch:%d/%d] [train_loss:%.4f] [valid_loss(RMSE):%.4f (Amp) + %.4f (Phase)= %.4f] [elapsed_time:%d:%2d:%2d] [Batch_time : %d:%d]" %
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
                
                
                zzz=0
                for nn in range( 8 ):
                    for nnn in range( 8 ):
                        amp_recon[  (nn*128) : (nn+1)*128 , (nnn*128) : (nnn+1)*128 ] = valid_pred_amp[pid*64 + zzz,:,:]
                        phase_recon[  (nn*128) : (nn+1)*128 , (nnn*128) : (nnn+1)*128 ] = valid_pred_phase[pid*64+zzz,:,:]
                        amp_valid_compare[  (nn*128) : (nn+1)*128 , (nnn*128) : (nnn+1)*128 ] = valid_amp[pid*64+zzz,:,:]
                        phase_valid_compare[  (nn*128) : (nn+1)*128 , (nnn*128) : (nnn+1)*128 ] = valid_phase[pid*64+zzz,:,:]
                        
                        zzz+=1
                
    
                
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
                      
                        
                fig.savefig('./Out_figure/epoch%d_%d_RMSE%.4f.png' %
                    ( epoch, samples_done, valid_loss), dpi = 150)
                plt.close()
                
                
                
                
            del valid_pred ,valid_pred_amp, valid_pred_phase
        del train_loss , train_data
        
        

    # Save model checkpoints      
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # model.save('SavedModels/%s_to_%s_epoch%d.hdf5' % (Settings["size_LR"] , Settings["size_HR"], epoch + 1))
        model.save('SavedModels/%s_to_%s_epoch%d_val_loss_%.4f.hdf5' % (Settings["size_LR"] , Settings["size_HR"], epoch + 1, valid_loss))


