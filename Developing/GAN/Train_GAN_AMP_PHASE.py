# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:19:39 2020

By pytorch version

@author: biosi
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import os

from glob import glob
import numpy as np
from numpy.random import randint
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from Models import  Generator_go , Discriminator_go
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from config import config_func
from Datasets import Datasets
from utils import gseq , Settings, indices, CTF_s, fftshift, ifftshift, n_iter , Phase_shift , FPM_Sampling
import pytorch_ssim 
import time



cuda = True if torch.cuda.is_available() else False

num_train  = 335
num_val  = 15 
num_test = 50  
num_data = num_train + num_val + num_test


opt = config_func()


Test_check_id = 1
N_iter = n_iter



dataset_train = Datasets('Train' , Settings)
dataset_valid = Datasets('Valid' , Settings)
dataset_test = Datasets('Test' , Settings)



BS = opt.batch_size

dataloader_train = DataLoader(dataset_train, batch_size=BS, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=BS, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=BS, shuffle=False)




dataset_check = Datasets('Check' , Settings)
chechec = DataLoader(dataset_check , batch_size = 1)
for i, (cover_range_show_ovr, samp_centers) in enumerate(chechec):
    print(i)




for i, (FPM_tests, HR_tsets, objectfunctions , objectFTs , _ , _ , FPM_tests_abs) in enumerate(dataloader_test):
    if i==0:
        test_FPM_sets = FPM_tests
        test_FPM_sets_abs = FPM_tests_abs
        
        test_HR_sets = HR_tsets
        
        test_objectfunction_sets = objectfunctions
        
        test_objectFT_sets = objectFTs
        break
    
    
# test_HR_sets=test_HR_sets.cpu().numpy()
# test_objectFT_sets=test_objectFT_sets.cpu().numpy()




for i, (vFPM_sets, vHR_sets, vobjectfunctions , vobjectFTs , _ , _ , vFPM_sets_abs) in enumerate(dataloader_valid):
    if i==0:
        valid_FPM_sets = vFPM_sets
        valid_FPM_sets_abs = vFPM_sets_abs
        valid_HR_sets = vHR_sets
        valid_objectfunction_sets = vobjectfunctions
        
        valid_objectFT_sets = vobjectFTs
        break

         

adv_loss = nn.BCELoss()
adv_loss2 = nn.BCELoss()
loss1 = nn.MSELoss()
loss2 = nn.MSELoss()
loss3 = nn.MSELoss()
loss4 = nn.MSELoss()
loss5 = nn.MSELoss()


generator = Generator_go(Settings['arraysize']**2, 2)
discriminator = Discriminator_go(2,1)

# discriminator_low = Discriminator_low(121,1)


if cuda:
    generator.cuda()
    discriminator.cuda()
    # discriminator_low.cuda()
    adv_loss.cuda()
    adv_loss2.cuda()
    loss1.cuda()
    loss2.cuda()
    loss3.cuda()
    loss4.cuda()
    loss5.cuda()
    
    
    
optimizer_G = torch.optim.Adam(generator.parameters() , lr=opt.lr) 
optimizer_D1 = torch.optim.Adam(discriminator.parameters() , lr=0.1*opt.lr) 
# optimizer_D2 = torch.optim.Adam(discriminator_low.parameters() , lr=opt.lr) 

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    




for epoch in range(opt.n_epochs):
    

    t_estart= time.time()
    for i, (FPM_sets, HR_sets, object_sets , object_FT_sets, _ , _ , FPM_sets_abs) in enumerate(dataloader_train):    
        
        
                
        valid = Variable(Tensor(FPM_sets.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(FPM_sets.size(0), 1).fill_(0.0), requires_grad=False)
        

        t_bstart = time.time()
        batch_size = object_FT_sets.shape[0]
        
        in_FPM_sets= Variable(FPM_sets )
    
        # -----------------
        #  Train Model
        # -----------------

        optimizer_G.zero_grad()
        
        GT_HR = Variable(torch.view_as_real(object_sets)).permute(0,3,1,2).cuda().type(torch.float32)
        
        
        
        
        pred_tr = generator(in_FPM_sets)
        
        # pred_tr_amp = torch.clamp( pred_tr[:,0,:,:], min =  1e-6)
        pred_tr_amp = pred_tr[:,0,:,:]
        GT_HR_amp = torch.clamp( torch.sqrt( GT_HR[:,0,:,:]**2 +GT_HR[:,1,:,:]**2  ) , min =  1e-6)
        
        # pred_tr_phase = torch.clamp( pred_tr[:,1,:,:],  min = -np.pi+ 1e-6, max = np.pi - 1e-6 )
        pred_tr_phase = pred_tr[:,1,:,:]
        GT_HR_phase = torch.clamp( torch.angle(  torch.view_as_complex(GT_HR.permute(0,2,3,1).contiguous())  )  , min =  -np.pi+ 1e-6 , max = np.pi - 1e-6 )
        
        
        # pred_tr_Kspace = fftshift( torch.fft(  pred_tr.permute(0,2,3,1)  ,2   ) ,[ -2, -3] )
        # GT_HR_Kspace = fftshift( torch.fft(  GT_HR.permute(0,2,3,1)  ,2   ) ,[ -2, -3] )
        
                
        # pred_tr_sampled , _= FPM_Sampling(pred_tr.permute(0,2,3,1), CTF_s , indices , Settings )
        # GT_HR_sampled , _= FPM_Sampling(GT_HR.permute(0,2,3,1), CTF_s , indices , Settings )
        
                
        for param in generator.parameters():
            reg_loss = param.norm(2)

        
        g_loss = adv_loss(discriminator(pred_tr) , valid)
        # g_loss = adv_loss(discriminator(pred_tr  ) , valid)
        # g_loss1 = adv_loss2(discriminator_low(pred_tr_sampled.permute(0,3,1,2) ) , valid)
        
        # ri_loss = loss1( pred_tr, GT_HR )
        amp_loss = loss1( pred_tr_amp,  GT_HR_amp )
        
        
        
        # amp_loss = loss2( pred_tr[:,0,:,:],  GT_HR_phase )
        
        
        
        phase_loss = loss2 ( pred_tr_phase , GT_HR_phase)
        # kspace_loss = 0.0001*loss4 (pred_tr_Kspace , GT_HR_Kspace )        
        # sampled_loss = 0.01*loss5(pred_tr_sampled ,GT_HR_sampled )
        
        
        # 
        ri_loss = 0
        # phase_loss = 0 
        kspace_loss = 0 
        sampled_loss = 0
        
        
        
        
        
        total_loss = amp_loss + 0.5*phase_loss + 0.05*g_loss +  50*reg_loss
        # total_loss =  g_loss
        # total_loss = ri_loss + 3*amp_loss + phase_loss + kspace_loss + 1.5*sampled_loss
        # g_loss=0
        # amp_loss.backward(retain_graph = True)
        total_loss.backward()
        # amp_loss.backward(retain_graph = True)
        # g_loss.backward()
        optimizer_G.step()
        
        
        ##################################################################################
        
        
        optimizer_D1.zero_grad()
        
                    
        for param in discriminator.parameters():
            reg_loss = param.norm(2)
            
        real_loss = adv_loss(discriminator(GT_HR ) , valid)
        fake_loss = adv_loss(discriminator(pred_tr.detach() ) , fake)
        
        d_loss = (real_loss + fake_loss)/2 + reg_loss
        
        
        d_loss.backward()
        
        optimizer_D1.step()
        
        
        ##################################################################################
        
        # optimizer_D2.zero_grad()
        
            
            
        # real_loss2 = adv_loss2(discriminator_low(GT_HR_sampled.permute(0,3,1,2) ) , valid)
        # fake_loss2 = adv_loss2(discriminator_low(pred_tr_sampled.detach().permute(0,3,1,2) ) , fake)
        
        # d_loss2 = (real_loss2 + fake_loss2)/2
        
        # d_loss2.backward()
        
        # optimizer_D2.step()
        
        #########################################################################
        #########################################################################
        
        
        
        t_bend = time.time()
        
        steps = 50
        if i% steps==0:
                        
            
            vvalid = Variable(Tensor(valid_FPM_sets.size(0), 1).fill_(1.0), requires_grad=False)
            vfake = Variable(Tensor(valid_FPM_sets.size(0), 1).fill_(0.0), requires_grad=False)
            
    
            
            vin_FPM_sets= Variable(valid_FPM_sets , requires_grad = True)
        
            # -----------------
            #  Train Model
            # -----------------
    
            vGT_HR = Variable(torch.view_as_real(valid_objectfunction_sets) , requires_grad = False).permute(0,3,1,2).cuda().type(torch.float32)
      
            pred_val = generator(vin_FPM_sets)
    
        
            vGT_HR_phase = torch.clamp( torch.angle(  torch.view_as_complex(vGT_HR.permute(0,2,3,1).contiguous())  )  , min =  -np.pi+ 1e-6 , max = np.pi - 1e-6 )
            
            # pred_val_amp = torch.clamp( torch.sqrt( pred_val[:,0,:,:]**2 +pred_val[:,1,:,:]**2  ) , min =  1e-6)
            # vGT_HR_amp = torch.clamp( torch.sqrt( vGT_HR[:,0,:,:]**2 +vGT_HR[:,1,:,:]**2  ) , min =  1e-6)
        
            # pred_val_phase = torch.clamp( torch.angle(  torch.view_as_complex(pred_val.permute(0,2,3,1).contiguous())  )  , min =  1e-6)
            # vGT_HR_phase = torch.clamp( torch.angle(  torch.view_as_complex(vGT_HR.permute(0,2,3,1).contiguous())  )  , min =  1e-6)
        
            
            # pred_val_Kspace = fftshift( torch.fft(  pred_val.permute(0,2,3,1)  ,2   ) ,[ -2, -3] )
            # vGT_HR_Kspace = fftshift( torch.fft(  vGT_HR.permute(0,2,3,1)  ,2   ) ,[ -2, -3] )
        
        
        
        
        
               
            
            
            #########################################################################
            #########################################################################
                    
        
            vg_loss = adv_loss(discriminator(pred_val ) , valid)
            
            
            
            
            vreal_loss = adv_loss(discriminator(vGT_HR ) , valid)
            vfake_loss = adv_loss(discriminator(pred_val ) , fake)
            
            vd_loss = (vreal_loss + vfake_loss)/2
            
            # g_loss=0
            # d_loss=0
            # vg_loss=0
            # vd_loss=0
            
            print( "[Epoch %d/%d] [Total_loss : %.3f(R) + %.3f(A) + %.3f(P) + %.3f(G) + %.3f(D) = %.3f ] [Batch : %d / %d] [Time : %.2f] [Val_g_loss ,Val_d_loss: %.3f , %.3f]"
                 % (epoch, opt.n_epochs, ri_loss ,amp_loss , phase_loss, g_loss , d_loss , total_loss ,  i+1, num_train * (1024  / Settings["size_HR"] )**2 / BS, (t_bend - t_bstart )*steps , vg_loss, vd_loss   ) )
              
      ############################################      
        if i% 500 ==0:

            
            tin_FPM_sets= Variable(test_FPM_sets , requires_grad = False)
        
            # -----------------
            #  Train Model
            # -----------------
    
            pred_test = generator(tin_FPM_sets).detach().cpu().numpy()
    
            # pred_obj_CNN1 =pred_test[Test_check_id,0,:,:] 
            # pred_obj_CNN1 = pred_test[Test_check_id,0,:,:] * np.exp( 1j*pred_test[Test_check_id,1,:,:])   
            
            
            pred_test_amp =  pred_test[Test_check_id,0,:,:]
            
            pred_test_phase = pred_test[Test_check_id,1,:,:]
        
        
        
            # pred_obj_CNN1 =pred_test_amp * np.exp(1j * pred_test_phase)
            
        
        
            test_object_complex = test_objectfunction_sets[Test_check_id, :,:].cpu().numpy()
            
            
            # (torch.angle(pred_obj_CNN1_t).max())
            # (torch.angle(pred_obj_CNN1_t).min())
            
            # np.min(pred_test[Test_check_id,0,:,:])
            # np.max(pred_test[Test_check_id,0,:,:])
            
            # np.min(np.angle(pred_obj_CNN1))
            # np.max(np.angle(pred_obj_CNN1))
            
            # plt.imshow(pred_test[Test_check_id,0,:,:])
            # plt.imshow(np.angle(pred_obj_CNN1) )
            # np.angle(1-0.1j)
            
            
            
            
            fig, axs = plt.subplots(4, 3, figsize = (50,40))

            fig.suptitle('Comparison_GAN ,  Epoch = %d , Batch = %d'  %(epoch,i) ,  fontsize=16)
           
        
        
        
        
            # axs[0, 0].imshow(np.absolute(pred_object_recover_comp  )                     ,  cmap='gray' , vmin = -0.05 , vmax=1.05) ;     axs[0, 0].axis('off');   axs[0, 0].set_title('AP_output_AMP' );                                                                                                         axs[0,0].title.set_size(15)
            # axs[1, 0].hist(np.absolute(pred_object_recover_comp).ravel(), bins=256 , range = (-0.05 , 1.05));  axs[1, 0].set_title( 'AP_output_AMP_Hist, Min = %.4f Max = %.4f' %((np.absolute( tobject_sets_shift_comp)).min() , (np.absolute( tobject_sets_shift_comp)).max())); 
            # axs[2, 0].imshow(np.angle(pred_object_recover_comp)               ,  cmap='gray', vmin = -3.14 , vmax=3.14) ;     axs[2, 0].axis('off');   axs[2, 0].set_title('AP_output_Phase %f , %f' %(np.min(np.angle( tobject_sets_shift_comp)) , np.max(np.angle( tobject_sets_shift_comp))));    axs[1,0].title.set_size(15)
            # axs[3, 0].hist(np.angle(pred_object_recover_comp).ravel(), bins=256 , range = (-3.14 , 3.14));     axs[3, 0].set_title( 'AP_output_phase_Hist, Min = %.4f Max = %.4f' %(np.min(np.angle( tobject_sets_shift_comp)) , np.max(np.angle( tobject_sets_shift_comp)))); 
            # axs[4, 0].axis("off");
        
        
            # axs[0, 1].imshow(np.absolute(pred_object_recover_shift_comp  )              ,  cmap='gray', vmin = -0.05 , vmax=1.05) ;            axs[0, 1].axis('off');   axs[0, 1].set_title('AMP_GT_shifted' );                                                                                                        axs[0,1].title.set_size(15)
            # axs[1, 1].hist(np.absolute(pred_object_recover_shift_comp).ravel(), bins=256 , range = (-0.05 , 1.05));  axs[1, 1].set_title( 'AP_output_AMP_shift_Hist, Min = %.4f Max = %.4f' %((np.absolute( pred_object_recover_shift_comp)).min() , (np.absolute( pred_object_recover_shift_comp)).max())); 
            # axs[2, 1].imshow(np.angle(pred_object_recover_shift_comp)                   ,  cmap='gray') ;            axs[2, 1].axis('off');   axs[2, 1].set_title('Phase_GT_shifted %f , %f' %(np.min(np.angle( pred_object_recover_shift_comp)) , np.max(np.angle( pred_object_recover_shift_comp))));    axs[2,1].title.set_size(15)
            # axs[3, 1].hist(np.angle(pred_object_recover_shift_comp).ravel(), bins=256 , range = (-3.14 , 3.14));     axs[3, 1].set_title( 'AP_output_phase_shift_Hist, Min = %.4f Max = %.4f' %(np.min(np.angle( pred_object_recover_shift_comp)) , np.max(np.angle( pred_object_recover_shift_comp))));  
            # axs[4, 1].axis("off");
            
           
            # axs[0, 0].imshow(pred_test[Test_check_id,0,:,:]                  ,  cmap='gray', vmin = -3.14 , vmax=3.14) ;     axs[0, 0].axis('off');   axs[0, 0].set_title('AMP_CNN' );                                                                                                        axs[0,0].title.set_size(15)
            # axs[1, 0].hist(pred_test[Test_check_id,0,:,:].ravel(), bins=256 , range = (-3.14 , 3.14));                            axs[1, 0].set_title( 'CNN_output_AMP_Hist, Min = %.4f Max = %.4f' %(pred_test[Test_check_id,0,:,:].min() , pred_test[Test_check_id,0,:,:].max())); 
            
            axs[0, 0].imshow(pred_test_amp                                    ,  cmap='gray', vmin = -0.05 , vmax=1.05) ;     axs[0, 0].axis('off');   axs[0, 0].set_title('AMP_CNN' );                                                                                                        axs[0,0].title.set_size(15)
            axs[1, 0].hist(pred_test_amp.ravel(), bins=256 , range = (-0.05 , 1.05));                            axs[1, 0].set_title( 'CNN_output_AMP_Hist, Min = %.4f Max = %.4f' %(pred_test_amp.min() , pred_test_amp.max())); 
            axs[2, 0].imshow(pred_test_phase                                 ,  cmap='gray', vmin = -3.14 , vmax=3.14) ;     axs[2, 0].axis('off');   axs[2, 0].set_title('Phase_CNN %f , %f' %(np.min(pred_test_phase) , np.max(pred_test_phase)));                        axs[2,0].title.set_size(15)
            axs[3, 0].hist(pred_test_phase.ravel(), bins=256 , range = (-3.14 , 3.14));                                  axs[3, 0].set_title( 'CNN_phase_shift_Hist, Min = %.4f Max = %.4f' %(np.min(pred_test_phase) , np.max(pred_test_phase)));  
            # axs[4, 1].hist(np.angle(pred_obj_CNN1).ravel(), bins=256 , range = (-3.14 , 3.14));     axs[4, 1].set_title( 'OverlapHist, Min = %.4f Max = %.4f' %(np.min(np.angle( pred_obj_CNN1)) , np.max(np.angle( pred_obj_CNN1))));  
            # axs[4, 2].hist(np.angle(tobject_sets_shift_comp).ravel(), bins=256 , range = (-3.14 , 3.14));     
            
            



            # axs[0, 3].imshow(np.absolute(tobject_sets_shift_comp  )                     ,  cmap='gray', vmin = -0.05 , vmax=1.05) ;     axs[0, 3].axis('off');   axs[0, 3].set_title('AMP_GT_shifted' );                                                                                                                  axs[0,3].title.set_size(15)
            # axs[1, 3].hist(np.absolute(tobject_sets_shift_comp).numpy().ravel(), bins=256 , range = (-0.05 , 1.05));  axs[1, 3].set_title( 'GT_AMP_shift_Hist, Min = %.4f Max = %.4f' %((np.absolute( tobject_sets_shift_comp)).min() , (np.absolute( tobject_sets_shift_comp)).max())); 
            # axs[2, 3].imshow(np.angle(tobject_sets_shift_comp)                          ,  cmap='gray', vmin = -3.14 , vmax=3.14) ;     axs[2, 3].axis('off');   axs[2, 3].set_title('Phase_GT_shifted %f , %f' %(np.min(np.angle( tobject_sets_shift_comp)) , np.max(np.angle( tobject_sets_shift_comp))));            axs[2,3].title.set_size(15)
            # axs[3, 3].hist(np.angle(tobject_sets_shift_comp).ravel(), bins=256 , range = (-3.14 , 3.14));     axs[3, 3].set_title( 'GT_phase_shift_Hist, Min = %.4f Max = %.4f' %(np.min(np.angle( tobject_sets_shift_comp)) , np.max(np.angle( tobject_sets_shift_comp))));  
            # axs[4, 3].axis("off");


            axs[0, 1].imshow(np.absolute(test_object_complex  )                     ,  cmap='gray', vmin = -0.05 , vmax=1.05) ;     axs[0, 1].axis('off');   axs[0, 1].set_title('AMP_original_shifted' );                                                                                                                  axs[0,1].title.set_size(15)
            axs[1, 1].hist(np.absolute(test_object_complex).ravel(), bins=256 , range = (-0.05 , 1.05));                            axs[1, 1].set_title( 'GT_AMP_shift_Hist, Min = %.4f Max = %.4f' %((np.absolute( test_object_complex)).min() , (np.absolute( test_object_complex)).max())); 
            axs[2, 1].imshow(np.angle(test_object_complex)                          ,  cmap='gray', vmin = -3.14 , vmax=3.14) ;     axs[2, 1].axis('off');   axs[2, 1].set_title('Phase_original_shifted %f , %f' %(np.min(np.angle( test_object_complex)) , np.max(np.angle( test_object_complex))));                        axs[2,1].title.set_size(15)
            axs[3, 1].hist(np.angle(test_object_complex).ravel(), bins=256 , range = (-3.14 , 3.14));                               axs[3, 1].set_title( 'GT_phase_Hist, Min = %.4f Max = %.4f' %(np.min(np.angle( test_object_complex)) , np.max(np.angle( test_object_complex))));  
           

           
           
                      
            
            axs[0, -1].imshow(cover_range_show_ovr[0,:,:], cmap = 'gray'); axs[0,-1].axis('off'), axs[0,-1].set_title('Cover_range');  axs[0,-1].scatter(samp_centers[0,:,0]  , samp_centers[0,:,1] , c='r' , s=10  );       axs[0,-1].title.set_size(20)
            axs[1, -1].axis("off"); axs[1,-1].invert_yaxis(); axs[1,-1].text( 0.1,0.3,['size_HR' ,Settings['size_HR']]  )
            axs[1, -1].axis("off"); axs[1,-1].invert_yaxis(); axs[1,-1].text( 0.1,0.4,['size_LR' ,Settings['size_LR']])
            axs[1, -1].axis("off"); axs[1,-1].invert_yaxis(); axs[1,-1].text( 0.1,0.5,['arraysize' ,Settings['arraysize']])
            axs[1, -1].axis("off"); axs[1,-1].invert_yaxis(); axs[1,-1].text( 0.1,0.6,['NA' ,Settings['NA']])
            axs[1, -1].axis("off"); axs[1,-1].invert_yaxis(); axs[1,-1].text( 0.1,0.7,['LEDgap' ,Settings['LEDgap']])
            axs[1, -1].axis("off"); axs[1,-1].invert_yaxis(); axs[1,-1].text( 0.1,0.8,['LEDheight' ,Settings['LEDheight']])
            
            axs[2, -1].axis("off"); axs[2,-1].invert_yaxis(); axs[2,-1].text( 0.1,0.3,['AMP_SSIM' ,pytorch_ssim.ssim(torch.tensor(pred_test_amp[None,None,:,:]).double() ,torch.tensor(np.absolute(test_object_complex )[None,None,:,:]).double())], fontsize = 35);   
            # axs[2, -1].axis("off"); axs[2,-1].invert_yaxis(); axs[2,-1].text( 0.1,0.6,['AMP_PSNR' ,Settings['LEDheight']])
            
            axs[3, -1].axis("off"); axs[1,-1].invert_yaxis(); axs[3,-1].text( 0.1,0.5,['PHASE_SSIM' ,pytorch_ssim.ssim(torch.tensor(pred_test_phase[None,None,:,:]).double() ,torch.tensor(np.angle(test_object_complex )[None,None,:,:]).double())], fontsize = 35) ;
            
            # pytorch_ssim.ssim(np.absolute(test_object_complex )[None,None,:,:] ,np.absolute(test_object_complex )[None,None,:,:])
            
# sss = pytorch_ssim.ssim(pred_test_amp[None,None,:,:] ,np.absolute(test_object_complex  )[None,None,:,:])
            fig.savefig('./Out_figure/GAN/epoch%d_%d_RMSE%.4f.png' %
                ( epoch, (i+1), vd_loss), dpi = 200)
            plt.close()
            
    PATH = './SavedModels/GAN/epoch_%d_val_loss_%f.pt' %(epoch , vd_loss)
    torch.save({
        'epoch': epoch,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer_G.state_dict(),
        'loss': d_loss,        }, PATH)


        
            
    t_eend = time.time()