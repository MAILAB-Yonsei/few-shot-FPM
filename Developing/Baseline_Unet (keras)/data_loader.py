from glob import glob
import numpy as np
from numpy.random import randint
import random
import scipy.io as sio
import matplotlib.pyplot as plt

class DataLoader():
        
    def __init__(self, dataset_name, mode):
  
        if mode == 'Check' :
            self.path_Amp = sorted( glob('../../Data/Amplitude_%s/Valid/*.*' % (dataset_name)  ))
            self.path_Phase = sorted( glob('../../Data/Phase_%s/Valid/*.*' % (dataset_name) ))
            
            # self.path_Comp = sorted(glob('../../../Data/Complex_%s/%s/*.*' % (dataset_name, mode)))
            
        else :
            self.path_Amp = sorted( glob('../../Data/Amplitude_%s/%s/*.*' % (dataset_name, mode)  ))
            self.path_Phase = sorted( glob('../../Data/Phase_%s/%s/*.*' % (dataset_name, mode) ))
    
            # self.path_Comp = sorted(glob('../../../Data/Complex_%s/%s/*.*' % (dataset_name, mode)))
        self.mode = mode
    
    
    def load_data(self, opt, Settings):
       
        amp = []; phase = []; FPM_datasets = []
        
        if self.mode == 'Train':
            data_num = opt.batch_size
            
        elif self.mode == 'Check':
            data_num = 1
            
        elif self.mode == 'Valid' or self.mode == 'Test':
            data_num = len(self.path_Amp)
            
        for i in range(data_num):                               ## regarding batch size
            if self.mode == 'Check' :
                ind_im  = i
                print("Check")
            
            elif self.mode == 'Valid' or self.mode == 'Test':
                ind_im = i
                # print("valid or test")
            else:
                ind_im = randint(0, len(self.path_Amp))
                # print("Training")
                
            # img_comp = sio.loadmat(self.path_Amp[ind_im])['img_comp']
            
            img_Amp   = sio.loadmat(self.path_Amp[ind_im])['img_split']
            img_Phase = sio.loadmat(self.path_Phase[ind_im])['img_split']
            
            if self.mode == 'Valid' or self.mode == 'Test':
                batch_random_index = sorted(random.sample(range(0,64) , 64))
                
            elif self.mode == 'Train':
                batch_random_index = sorted(random.sample(range(0,64) , 8))
                
            elif self.mode == 'Check':
                batch_random_index=  [0]
                cover_range =  np.ones(( Settings["size_HR"] , Settings["size_HR"]))
                                
                cover_range_show =  np.zeros(( Settings["size_HR"] , Settings["size_HR"]))
                cover_range_show_cum =  np.zeros(( Settings["size_HR"] , Settings["size_HR"]))
                cover_range_show_ovr =  np.zeros(( Settings["size_HR"] , Settings["size_HR"]))
                cover_range_show_ctr =  np.zeros(( Settings["size_HR"] , Settings["size_HR"]))
                samp_centers = np.zeros((Settings["arraysize"]**2, 2))
                            
            for c in (batch_random_index):
                object_function = np.multiply(img_Amp[:,:,c],np.exp(2j*np.pi*img_Phase[:,:,c]))
                
                # print(object_function.shape, 'object_function_shape')
                ###################################################################
                
                waveLength = 0.518e-6;    #사용된 빛의 종류
                k0 = 2*np.pi/waveLength;  # wavevector
        
                #parameter for camera
                
                #parameter for simulation (Ground Truth)
                psize = Settings["spsize"] / ( Settings["size_HR"] / Settings["size_LR"]);

                xlocation = np.zeros((1,Settings["arraysize"]**2));
                ylocation = np.zeros((1,Settings["arraysize"]**2));
                        
                for it in range (0,Settings["arraysize"]):# from top left to bottom right
                    for k in range(0,Settings["arraysize"]):
                        xlocation[0, k + Settings["arraysize"]*(it)] = (((1 - Settings["arraysize"])/2) + k) * Settings["LEDgap"]
                        ylocation[0, k + Settings["arraysize"]*(it)] = ((Settings["arraysize"] - 1)/2 - (it)) * Settings["LEDgap"]
                

                kx_relative = -np.sin(np.arctan(xlocation/Settings["LEDheight"]));
                ky_relative = -np.sin(np.arctan(ylocation/Settings["LEDheight"]));
                
                [m,n,zmm] = img_Phase.shape;
                
                m1 = int(m/(Settings["spsize"]/psize)) # image size of the final output
                n1 = int(n/(Settings["spsize"]/psize))  
            
                imSeqLowRes = np.zeros((m1, n1, Settings["arraysize"]**2)); #the final low resolution image sequence
                kx = k0 * kx_relative;                 
                ky = k0 * ky_relative;
                dkx = 2*np.pi/(psize*n);
                dky = 2*np.pi/(psize*m);
                cutoffFrequency = Settings["NA"] * k0;
                kmax = np.pi/Settings["spsize"]
                kxm=np.zeros((m1,n1))
                kym=np.zeros((m1,n1))
                
                instx=np.linspace(-kmax,kmax,m1)
                insty=np.linspace(-kmax,kmax,n1)
                [kxm, kym]= np.meshgrid(instx,insty)
                CTF = ((kxm**2+kym**2) < cutoffFrequency**2); #pupil function circ(kmax);
                
                # plt.figure()
                # plt.imshow(CTF,cmap='gray')
                objectFT = np.fft.fftshift(np.fft.fft2(object_function));
                
                # print(objectFT.shape, 'ObjectFTs.shape')
                            
                ## Low Resolution 묶음 만들기
                for tt in range (0,Settings["arraysize"]**2):
                    kxc = int((n+1)/2+kx[0,tt]/dkx)
                    kyc = int((m+1)/2+ky[0,tt]/dky)
                    kxl=int((kxc-(n1-1)/2))
                    kxh=int((kxc+(n1-1)/2))
                    kyl=int((kyc-(m1-1)/2))
                    kyh=int((kyc+(m1-1)/2))
      
                    imSeqLowFT = objectFT[kyl:kyh+1,kxl:kxh+1]* CTF
                    
                    # plt.imshow(imSeqLowFT , cmap = 'gray')
                    
                    imSeqLowRes[:,:,tt] = np.absolute(np.fft.ifft2(np.fft.ifftshift(imSeqLowFT)))**2
                
                if self.mode == 'Check' :
                    
                    for tt in range (0,Settings["arraysize"]**2):
                        kxc = int((n+1)/2+kx[0,tt]/dkx)
                        kyc = int((m+1)/2+ky[0,tt]/dky)
                        kxl=int((kxc-(n1-1)/2))
                        kxh=int((kxc+(n1-1)/2))
                        kyl=int((kyc-(m1-1)/2))
                        kyh=int((kyc+(m1-1)/2))
                        
                        cover_range_show[kyl:kyh+1,kxl:kxh+1] += cover_range[kyl:kyh+1,kxl:kxh+1]* CTF
                        cover_range_show_cum[kyl:kyh+1,kxl:kxh+1] += cover_range[kyl:kyh+1,kxl:kxh+1]* CTF
                        samp_centers[tt,0] = kxc
                        samp_centers[tt,1] = kyc
                        plt.figure()
                        plt.imshow(cover_range_show, cmap = 'gray')
                        
                        if tt==0 or tt==1:
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
                    plt.scatter(samp_centers[:,0]  , samp_centers[:,1], c='r', s=10)
                    
                    plt.figure()
                    plt.imshow(CTF, cmap = 'gray')
                
                FPM_datasets.append(imSeqLowRes)
                amp.append(img_Amp[:,:,c])
                phase.append(img_Phase[:,:,c])
            
        FPM_datasets = np.array(FPM_datasets)
        amp = np.array(amp)
        phase = np.array(phase)
        
        amp=amp[:,:,:,np.newaxis]
        phase=phase[:,:,:,np.newaxis]
                        
        return {'FPM_datasets':FPM_datasets, 'amp':amp,
                'phase':phase}
    
    def len_data(self):
        return len(self.path_Amp)