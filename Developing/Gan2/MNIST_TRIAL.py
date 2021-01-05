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
from Models import AP,  AP_TJ, Phase_shift_UV , Generator_MNIST , Discriminator_go , Discriminator_MNIST
# from momo import weights_init_normal
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from config import config_func
from Datasets import Datasets
from utils import gseq , Settings, indices, CTF_s, fftshift, ifftshift, n_iter , Phase_shift , FPM_Sampling


import time
import matplotlib.pyplot as plt
from mnist import MNIST
import idx2numpy


train_image ='C:/Users/biosi/Desktop/FPM/Developing_pytorch/dataset_mnist/train-images.idx3-ubyte'
train_label ='C:/Users/biosi/Desktop/FPM/Developing_pytorch/dataset_mnist/train-labels.idx1-ubyte'
test_image ='C:/Users/biosi/Desktop/FPM/Developing_pytorch/dataset_mnist/t10k-images.idx3-ubyte'
test_label ='C:/Users/biosi/Desktop/FPM/Developing_pytorch/dataset_mnist/t10k-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(train_image)
train_labels = idx2numpy.convert_from_file(train_label)
test_images = idx2numpy.convert_from_file(test_image)
test_labels = idx2numpy.convert_from_file(test_label)




cuda = True if torch.cuda.is_available() else False



opt = config_func()





adv_loss = nn.BCELoss()
adv_loss2 = nn.BCELoss()
loss1 = nn.MSELoss()
loss2 = nn.MSELoss()
loss3 = nn.MSELoss()
loss4 = nn.MSELoss()
loss5 = nn.MSELoss()


generator = Generator_MNIST(1, 1)
discriminator = Discriminator_MNIST(1,1)

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
optimizer_D1 = torch.optim.Adam(discriminator.parameters() , lr=opt.lr) 
# optimizer_D2 = torch.optim.Adam(discriminator_low.parameters() , lr=opt.lr) 

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    




for epoch in range(opt.n_epochs):
    

    t_estart= time.time()
    for i, image in enumerate(train_images):    

                
        valid = Variable(Tensor(1, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(1, 1).fill_(0.0), requires_grad=False)
        

        t_bstart = time.time()
        batch_size = 1
        
        
        inputs_randoms = Variable( Tensor(np.random.rand(28,28)) )[None,None,:,:]
    
        # -----------------
        #  Train Model
        # -----------------

        optimizer_G.zero_grad()
        
        GT_HR = Variable(Tensor(image)).type(torch.float32)[None,None,:,:]
        
        
        
        
        pred_tr = generator(inputs_randoms)
        
         
        g_loss = adv_loss(discriminator(pred_tr  ) , valid)
        
        
      
        
        
        
        g_loss.backward()
        optimizer_G.step()
        
        
        ##################################################################################
        
        
        optimizer_D1.zero_grad()
        
            
            
        real_loss = adv_loss(discriminator(GT_HR ) , valid)
        fake_loss = adv_loss(discriminator(pred_tr.detach() ) , fake)
        
        d_loss = (real_loss + fake_loss)/2
        
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
   
            
        if i % 20==0:
        
            
            print( "[Epoch %d/%d] [loss : %.3f(G) + %.3f(D)] [Batch : %d / %d] "
                 % (epoch, opt.n_epochs,  g_loss , d_loss ,   i+1, 0   ) )
              
      ############################################      
        if i% 100 ==0:

            tinputs_randoms = Variable( Tensor(np.random.rand(28,28)) )[None,None,:,:]
    
        
            # -----------------
            #  Train Model
            # -----------------
    
            pred_test = generator(tinputs_randoms).detach().cpu().numpy()
    
            # pred_obj_CNN1 =pred_test[Test_check_id,0,:,:] 
            # pred_obj_CNN1 = pred_test[Test_check_id,0,:,:] * np.exp( 1j*pred_test[Test_check_id,1,:,:])   
            
            
        
            
            
            
            fig, axs = plt.subplots(2, 1, figsize = (50,40))

            fig.suptitle('Comparison_GAN ,  Epoch = %d , Batch = %d'  %(epoch,i) ,  fontsize=16)
           
        
            
            axs[0].imshow(pred_test[0,0,:,:], cmap = 'gray'); axs[0].axis('off')
            axs[1].imshow(GT_HR[0,0,:,:].detach().cpu(), cmap = 'gray'); axs[1].axis('off')
            
            
            
            fig.savefig('./Out_figure/MNIST/epoch%d_%d_RMSE%.4f.png' %
                ( epoch, (i+1), 0), dpi = 200)
            plt.close()
        
        
            
    t_eend = time.time()