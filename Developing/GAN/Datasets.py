import torch.nn as nn
import torch.nn.functional as F
import os

import torch as torch
from glob import glob
from torch.utils.data import Dataset, DataLoader

from config import config_func
from numpy.random import randint
import scipy.io as sio
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import gseq
from utils import Settings

opt = config_func()

sseq= gseq(Settings["arraysize"])


class Datasets(Dataset):
    
    
    def __init__(self,  mode, Settings):
  
        # if mode == 'Check' :
        #     self.path_Complex = sorted( glob('../Data/DIV2K_800_r1024_d64/Valid/0336_001.mat'  ))
        
        # else :
        #     self.path_Complex = sorted( glob('../Data/DIV2K_800_r1024_d64/%s/*.*' %  mode  ))
            
        # self.mode = mode
        
        
        if mode == 'Check' :
            self.path_Complex = sorted( glob('../Data/DIV2K_800_r1024_d128/Valid/0336_001.mat'  ))
        
        else :
            self.path_Complex = sorted( glob('../Data/DIV2K_800_r1024_d128/%s/*.*' %  mode  ))
            
        self.mode = mode
        
        
        
        
    def __getitem__(self, idx): 
        
            
        if self.mode == 'Check' :
            
            print("Check")
        
        
        img_Comp = sio.loadmat(self.path_Complex[idx])['img']
    
        
        
        img_Amp = np.clip( np.absolute(img_Comp) , 1e-3, 1)
        img_Phase = np.angle(img_Comp)
        
       
        
        if self.mode == 'Check':
            
            cover_range =  np.ones(( Settings["size_HR"] , Settings["size_HR"]    ))
                            
            cover_range_show =  np.zeros(( Settings["size_HR"] , Settings["size_HR"]    ))
            cover_range_show_cum =  np.zeros(( Settings["size_HR"] , Settings["size_HR"]    ))
            cover_range_show_ovr =  np.zeros(( Settings["size_HR"] , Settings["size_HR"]    ))
            cover_range_show_ctr =  np.zeros(( Settings["size_HR"] , Settings["size_HR"]    ))
            samp_centers = np.zeros((  Settings["arraysize"]**2, 2 ))
            
            # print(cover_range_FT.shape , "cover_Range_fT.shape")
            
                                     
            
            
        object_function = img_Comp
        
        # print(object_function.shape, 'object_function_shape')
 
        ###################################################################
        
        waveLength = 0.518e-6;    #사용된 빛의 종류
        k0 = 2*np.pi/waveLength;  # wavevector

        #parameter for camera
        spsize = Settings["Spsize"]       #sampling pixel size of the camera
        #parameter for simulation (Ground Truth)
        psize = spsize / ( Settings["size_HR"] / Settings["size_LR"]);
        

  
        xlocation = np.zeros((1,Settings["arraysize"]**2));
        ylocation = np.zeros((1,Settings["arraysize"]**2));
                
        for it in range (0,Settings["arraysize"]):# from top left to bottom right
            for k in range(0,Settings["arraysize"]):
                xlocation[0, k + Settings["arraysize"]*(it)] = (((1 - Settings["arraysize"])/2) + k) * Settings["LEDgap"]
                ylocation[0, k + Settings["arraysize"]*(it)] = ((Settings["arraysize"] - 1)/2 - (it)) * Settings["LEDgap"]
        
        
        
        
        kx_relative = -np.sin(np.arctan(xlocation/Settings["LEDheight"]));
        ky_relative = -np.sin(np.arctan(ylocation/Settings["LEDheight"]));
        
        
        [m,n] = img_Phase.shape;
        
        
        m1 = int(m/(spsize/psize))     #image size of the final output
        n1 = int(n/(spsize/psize))  
    
                
        imSeqLowRes = np.zeros((m1, n1, Settings["arraysize"]**2)); #the final low resolution image sequence
        imSeqLowRes_abs = np.zeros((m1, n1, Settings["arraysize"]**2)); #the final low resolution image sequence
        imSeqLowResPhase = np.zeros((m1, n1, Settings["arraysize"]**2)); #the final low resolution image sequence
        imSeqLowResFT = np.zeros((m1, n1, Settings["arraysize"]**2) , dtype= np.complex64); # FT of the final low resolution image sequence (complex)
        kx = k0 * kx_relative;                 
        ky = k0 * ky_relative;
        dkx = 2*np.pi/(psize*n);
        dky = 2*np.pi/(psize*m);
        cutoffFrequency = Settings["NA"] * k0;
        kmax = np.pi/spsize
        kxm=np.zeros((m1,n1))
        kym=np.zeros((m1,n1))
        
        
        instx=np.linspace(-kmax,kmax,m1)
        insty=np.linspace(-kmax,kmax,n1)
        [kxm, kym]= np.meshgrid(instx,insty)
        CTF = ((kxm**2+kym**2) < cutoffFrequency**2); #pupil function circ(kmax);
        
        # plt.figure()
        # plt.imshow(CTF,cmap='gray')
     
        
        objectFT = np.fft.fftshift(np.fft.fft2(object_function));
        
        
        # print(objectFT.dtype)
        # print(objectFT.shape, 'ObjectFTs.shape')
        
     


        
        ## Low Resolution 묶음 만들기
        for tt in range (0,Settings["arraysize"]**2):
            
            ittt = int(sseq[tt]-1)
            # ittt = int(sseq[Settings["arraysize"]**2-1-tt]-1)   ### reverse
    
            
            kxc = int((n+1)/2+kx[0,ittt]/dkx)
            kyc = int((m+1)/2+ky[0,ittt]/dky)
            kxl=int((kxc-(n1-1)/2))
            kxh=int((kxc+(n1-1)/2))
            kyl=int((kyc-(m1-1)/2))
            kyh=int((kyc+(m1-1)/2))
            
   
            
            imSeqLowFT = objectFT[kyl:kyh+1,kxl:kxh+1]* CTF
            
            imSeqLowResFT[:,:,tt] = imSeqLowFT
            # plt.imshow(imSeqLowFT , cmap = 'gray')
            
            imSeqLowRes[:,:,tt] = np.absolute(np.fft.ifft2(np.fft.ifftshift(imSeqLowFT)))**2
            imSeqLowRes_abs[:,:,tt] = np.absolute(np.fft.ifft2(np.fft.ifftshift(imSeqLowFT)))
            imSeqLowResPhase[:,:,tt] = np.angle(np.fft.ifft2(np.fft.ifftshift(imSeqLowFT)))
        
        
        
        
        if self.mode == 'Check' :
            
            for tt in range (0,Settings["arraysize"]**2):
                
                ittt = int(sseq[tt]-1)
                # ittt = int(sseq[Settings["arraysize"]**2-1-tt]-1)   ### reverse 
                
                kxc = int((n+1)/2+kx[0,ittt]/dkx)
                kyc = int((m+1)/2+ky[0,ittt]/dky)
                kxl=int((kxc-(n1-1)/2))
                kxh=int((kxc+(n1-1)/2))
                kyl=int((kyc-(m1-1)/2))
                kyh=int((kyc+(m1-1)/2))
                
                
                cover_range_show[kyl:kyh+1,kxl:kxh+1] += cover_range[kyl:kyh+1,kxl:kxh+1]* CTF
                cover_range_show_cum[kyl:kyh+1,kxl:kxh+1] += cover_range[kyl:kyh+1,kxl:kxh+1]* CTF
                samp_centers[ittt,0] = kxc
                samp_centers[ittt,1] = kyc
                plt.figure()	
                plt.imshow(cover_range_show, cmap = 'gray')
                
                
                if ittt==0 or ittt==1:
                    
                    cover_range_show_ovr[kyl:kyh+1,kxl:kxh+1] += cover_range[kyl:kyh+1,kxl:kxh+1]* CTF
                # if tt==1:
                #     cover_range_show[kyl:kyh+1,kxl:kxh+1] += cover_range[kyl:kyh+1,kxl:kxh+1]* CTF
                
                cover_range_show[cover_range_show>0]=1
                
            plt.figure()
            plt.imshow(cover_range_show, cmap = 'gray')
            
            
            plt.figure()
            plt.imshow(cover_range_show_cum, cmap = 'gray')
            
            
            plt.figure()
            plt.imshow(cover_range_show_ovr, cmap = 'gray')
            
            
            plt.figure()
            plt.imshow(cover_range_show_ctr, cmap = 'gray')
            plt.scatter(samp_centers[:,0]  , samp_centers[:,1] , c='r' , s=10 )
            
            
            plt.figure()
            plt.imshow(CTF, cmap = 'gray')
            
            
            
            return cover_range_show_cum ,samp_centers
                  
                
           
            
            
            
        self.FPM_datasets = np.array(imSeqLowRes)
        self.FPM_datasets = np.transpose(self.FPM_datasets , (2, 0,1 ))
        
        self.FPM_datasets_abs = np.array(imSeqLowRes_abs)
        self.FPM_datasets_abs = np.transpose(self.FPM_datasets_abs , (2, 0,1 ))
      
        
        self.amp  = img_Amp[:,:,np.newaxis]
        self.phase =  img_Phase[:,:,np.newaxis]
        
        self.Out = np.concatenate( [self.amp,  self.phase] , axis=-1)
        self.Out = np.transpose(self.Out , (2,0,1 ))
              
            
        self.object_function = np.array(object_function)
        
        # self.object_functions = np.transpose(self.object_functions , (0, 3, 1,2 ))
        
            
        self.object_FT = np.array(objectFT)
        # print(self.object_FTs.shape, 'FT.shape')
        # self.object_FTs = np.transpose(self.object_FTs , (0, 3, 1,2 ))
        
        
        self.Low_FT_sets = np.array(imSeqLowResFT)
        self.Low_FT_sets = np.transpose(self.Low_FT_sets , (2,0,1 ))
    
        
        self.Low_Phase_sets = np.array(imSeqLowResPhase)
        self.Low_Phase_sets = np.transpose(self.Low_Phase_sets , (2, 0, 1))
    
        x = torch.cuda.FloatTensor(self.FPM_datasets)
        
        y = torch.cuda.FloatTensor(self.Out)
       
        z = self.object_function
        # z = torch.cuda.FloatTensor(self.object_functions[idx])
        
        
        d = self.object_FT
        # d = torch.cuda.FloatTensor(self.object_FTs[idx])
        f=  self.Low_FT_sets
        
        p = self.Low_Phase_sets
        
        a = torch.cuda.FloatTensor(self.FPM_datasets_abs)
        
        return x, y, z, d, f, p, a 
        
    def __len__(self):
        # return len(self.path_Amp)
        # return 1
        return len(self.path_Complex)
            
    
