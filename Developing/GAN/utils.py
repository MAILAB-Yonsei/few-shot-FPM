# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 01:14:50 2020

@author: biosi
"""
import numpy as np
import torch

from config import config_func
opt = config_func()

def gseq(arraysize):

    n=(arraysize+1)/2;
    sequence=np.zeros((2,arraysize**2))
    sequence[0,0]=n;
    sequence[1,0]=n;			
    dx=+1;
    dy=-1;
    stepx=+1;
    stepy=-1;
    direction=+1;
    counter=0;
  
    for i in range (1,arraysize**2):
        counter=counter+1;
        if (direction==+1):
            sequence[0,i]=sequence[0,i-1]+dx;
            sequence[1,i]=sequence[1,i-1];
            if (counter==abs(stepx)):
                counter=0;
                direction=direction*-1;
                dx=dx*-1;
                stepx=stepx*-1;
                if stepx>0:
                    stepx=stepx+1;
                else:
                    stepx=stepx-1;
    
        else:
            sequence[0,i]=sequence[0,i-1];
            sequence[1,i]=sequence[1,i-1]+dy;
            if (counter==abs(stepy)):
                counter=0;
                direction=direction*-1;
                dy=dy*-1;
                stepy=stepy*-1;
                if (stepy>0):
                    stepy=stepy+1;
                else:
                    stepy=stepy-1;
    
    seq=(sequence[0,:]-1)*arraysize+sequence[1,:]
    return seq

 

Settings = {
    "size_HR"    : 128 ,
    "size_LR"    : 32  ,
   
    "arraysize"  : 11  ,                #   5 // 7 // 9 /11
     "seq"       : gseq(11)-1,          
    #parameter for Imaging System
    "NA"         : 0.12    ,          ## 0.15 // 0.12   // 0.12 // 0.12
    "LEDgap"     : 4.5  ,             ## 12 // 7    // 6 // 4
    "LEDheight"  :  50   ,            ## 50  //  50   // 50  // 50
    "Spsize"   : 1.8e-6,               ## 1.5e-6 //  1.8e-6   // 1.8e-6 // 
}


n_iter = 10


m = Settings["size_HR"] 
n = Settings["size_HR"]

spsize = Settings["Spsize"]   
psize = Settings["Spsize"] / (Settings["size_HR"]/Settings["size_LR"]);

m1 = int(m/(Settings["Spsize"]/psize))    
n1 = int(n/(Settings["Spsize"]/psize))    

waveLength = 0.518e-6;    #사용된 빛의 종류
k0 = 2*np.pi/waveLength;  # wavevector
dkx = 2*np.pi/(psize*n);
dky = 2*np.pi/(psize*m);



xlocation = np.zeros((1,Settings["arraysize"]**2));
ylocation = np.zeros((1,Settings["arraysize"]**2));

for i in range (0,Settings["arraysize"]):# from top left to bottom right
    for k in range(0,Settings["arraysize"]):
        
        xlocation[0, k + Settings["arraysize"]*(i)] = (((1 - Settings["arraysize"])/2) + k) * Settings["LEDgap"]
        ylocation[0, k + Settings["arraysize"]*(i)] = ((Settings["arraysize"] - 1)/2 - (i)) * Settings["LEDgap"]
    
    
## Relative Fourier Domain location of LED array
kx_relative = -np.sin(np.arctan(xlocation/Settings["LEDheight"]));  
ky_relative = -np.sin(np.arctan(ylocation/Settings["LEDheight"]));   

kx = k0 * kx_relative;                 
ky = k0 * ky_relative;
    
seq=gseq(Settings["arraysize"])
objectRecover = np.ones((Settings["size_HR"],Settings["size_HR"]));
objectRecoverFT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(objectRecover)));

cutoffFrequency = Settings["NA"] * k0;
kmax = np.pi/Settings["Spsize"]
kxm=np.zeros((m1,n1))
kym=np.zeros((m1,n1))

instx=np.linspace(-kmax,kmax,m1)
insty=np.linspace(-kmax,kmax,n1)
[kxm, kym]= np.meshgrid(instx,insty)
CTF = ((kxm**2+kym**2) < cutoffFrequency**2); #pupil function circ(kmax);



CTF = torch.tensor(CTF ,dtype= torch.int32).cuda()

BS = opt.batch_size
CTF_s = torch.zeros((BS,Settings["size_LR"],Settings["size_LR"],2) ,dtype = torch.int32, device = 'cuda')
for k1 in range(BS):
    for k4 in range(2):
        CTF_s[k1,:,:,k4]=CTF



kyl_L = np.zeros((Settings["arraysize"]**2))
kyh_L = np.zeros((Settings["arraysize"]**2))
kxl_L = np.zeros((Settings["arraysize"]**2))
kxh_L = np.zeros((Settings["arraysize"]**2))


for i3 in range(0,Settings["arraysize"]**2):      ## first filling of the kspace
    i2=int(seq[i3]-1)  ### from center
    # i2=int(seq[Settings["arraysize"]**2-1-i3]-1)   ### reverse
    
    kxc = int((n+1)/2 + kx[0, i2]/dkx);
    kyc = int((m+1)/2 + ky[0, i2]/dky);
    kxl_L[i3] = int((kxc - (n1-1)/2))
    kxh_L[i3] = int((kxc + (n1-1)/2))
    kyl_L[i3] = int((kyc - (m1-1)/2))
    kyh_L[i3] = int((kyc + (m1-1)/2))





# kyl_S = np.zeros((Settings["arraysize"]**2))
# kyh_S = np.zeros((Settings["arraysize"]**2))
# kxl_S = np.zeros((Settings["arraysize"]**2))
# kxh_S = np.zeros((Settings["arraysize"]**2))
   
# for i4 in range(0,Settings["arraysize"]**2):      ## first filling of the kspace
       
#     kxc = int((n+1)/2 + kx[0, i4]/dkx);
#     kyc = int((m+1)/2 + ky[0, i4]/dky);
#     kxl_S[i4] = int((kxc - (n1-1)/2))
#     kxh_S[i4] = int((kxc + (n1-1)/2))
#     kyl_S[i4] = int((kyc - (m1-1)/2))
#     kyh_S[i4] = int((kyc + (m1-1)/2))



kxl = torch.cuda.IntTensor(kxl_L )
kxh = torch.cuda.IntTensor(kxh_L )
kyl = torch.cuda.IntTensor(kyl_L )
kyh = torch.cuda.IntTensor(kyh_L )



indices = {
    "kxl"    : kxl ,
    "kxh"    : kxh ,
    "kyl"    : kyl ,
    "kyh"    : kyh ,
    }


# kxl_S = torch.cuda.IntTensor(kxl_S )
# kxh_S = torch.cuda.IntTensor(kxh_S )
# kyl_S = torch.cuda.IntTensor(kyl_S )
# kyh_S = torch.cuda.IntTensor(kyh_S )

CTF_s   = torch.cuda.IntTensor(CTF_s )

####################################

# t= test_objectFT_sets[0,:,:]



# to = torch.tensor(test_objectfunction_sets[0,:,:])

# plt.imshow(torch.absolute(to))
# plt.imshow(torch.angle(to))


# ti = fftshift(torch.fft(torch.view_as_real(to ),2) )



# plt.imshow(abs(t))


# # t.shape

# plt.imshow  (  abs   (   np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(t) )      )))

# plt.imshow  (  np.angle (   np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(t) )      )))


# ttt= torch.tensor(t)
# plt.imshow(abs(ttt))
# torch.ifft(  torch.view_as_real(ifftshift(ttt))  , 2).dtype
# ttt.dtype

# tti = (   fftshift( torch.view_as_complex( torch.ifft(  torch.view_as_real(  ifftshift(ttt)  )  , 2)       ) ))
# plt.imshow(abs(tti))
# plt.imshow(torch.angle(tti))



# plt.imshow( torch.absolute(   torch.view_as_complex(ti)))

# ######################




def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)



def Phase_shift(AP_output) :
    
    
    
    AP_output_U = torch.div(AP_output , torch.clamp(   torch.sqrt(AP_output[:,:,:,0]**2 + AP_output[:,:,:,1]**2 )  , min=1e-6 )[:,:,:,None])
    
    AP_output_mean = torch.sum(AP_output_U , dim=[-2, -3])/(AP_output_U.shape[-2]*AP_output_U.shape[-3])
    
    
    AP_output_mean_unit= torch.div( AP_output_mean ,  torch.clamp(  torch.sqrt(AP_output_mean[:,0]**2 + AP_output_mean[:,1]**2) , min = 1e-6)[:,None]  )
    
    
    
    AP_rotated = torch.zeros(AP_output.shape, device='cuda')
    
    for z in range( AP_output_mean.shape[0]):
        # if AP_output_mean[z, 1] >0 :
        AP_rotated[z,:,:,0] =  AP_output[z,:,:,0]*AP_output_mean_unit[z,0] + AP_output[z,:,:,1]*AP_output_mean_unit[z,1]
        AP_rotated[z,:,:,1] =  AP_output[z,:,:,0]*AP_output_mean_unit[z,1] - AP_output[z,:,:,1]*AP_output_mean_unit[z,0]
            
        # else:
        #     AP_rotated[z,:,:,0] =  AP_output[z,:,:,0]*AP_output_mean_unit[z,0] - AP_output[z,:,:,1]*AP_output_mean_unit[z,1]  
        #     AP_rotated[z,:,:,1] =  AP_output[z,:,:,0]*AP_output_mean_unit[z,1] + AP_output[z,:,:,1]*AP_output_mean_unit[z,0]  
    
    
    
    return AP_rotated, AP_output_mean_unit 





def FPM_Sampling(RI_prediction , CTF_S , indicess, Settings):
    
    # obj_comp = torch.view_as_complex(RI_prediction)
    
    
    
    imSeqLowRes = torch.zeros( ( RI_prediction.shape[0], 32,32 ,Settings["arraysize"]**2) , device = 'cuda' )
    imSeqLowRes_abs = torch.zeros( ( RI_prediction.shape[0], 32,32,Settings["arraysize"]**2 ), device = 'cuda' )
    
    
    
    
    objectFT = fftshift(torch.fft(RI_prediction, 2),[-2,-3])    
    
    
    
    for tt in range (0,Settings["arraysize"]**2):
        
        kyl = indicess['kyl'][tt]
        kyh = indicess['kyh'][tt]
        kxl = indicess['kxl'][tt]
        kxh = indicess['kxh'][tt]
    
    
    
        imSeqLowFT = objectFT[:,kyl:kyh+1,kxl:kxh+1,:]* CTF_S
        
        
        
        imseqlowrer = torch.ifft( ifftshift(imSeqLowFT,[-2, -3]) , 2)

        imSeqLowRes[:,:,:,tt] = torch.clamp( imseqlowrer[:,:,:,0]**2 +imseqlowrer[:,:,:,1]**2  , min= 1e-6   )
        imSeqLowRes_abs[:,:,:,tt] = torch.clamp( torch.sqrt( imseqlowrer[:,:,:,0]**2 +imseqlowrer[:,:,:,1]**2  ), min= 1e-6   )
        
        # imSeqLowRes[:,:,:,tt] = fftshift(torch.fft
        
        
    
    
    # imSeqLowRes = torch.absolute(
    
        
        
        
    return imSeqLowRes , imSeqLowRes_abs
    
    
    
    
    
    
    
    
    
    
    
    


# A= test_HR_sets[:,:,:,0:3]

# A = np.transpose( A , [3,0,1,2])

# A.shape

# plt.imshow(A[0,:,:,0])
# plt.imshow(A[0,:,:,1])


# comp_object = np.multiply(  A[:,:,:,0] ,  np.exp(2j*np.pi*(   A[:,:,:,1]  -0.5)))  #### Complex domain ( real  - imag 1 channel)
                

# comp_object.shape

# plt.imshow(abs(comp_object[0,:,:]))                                     ### Complex object
# plt.imshow(np.angle(comp_object[0,:,:]))                                ###  complex object


# plt.imshow ( abs(np.fft.fft2(   comp_object ) ))                  ## complex object fftshift
# plt.imshow ( np.angle(np.fft.fft2(   comp_object ) ))              ## complex object fftshift 


# plt.imshow (   abs(   np.fft.ifftshift(comp_object)       ))                  ## complex object ifftshift        
# plt.imshow (   np.angle(      np.fft.ifftshift(comp_object)            ))     ## complex object ifftshift


# plt.imshow ( abs(    np.fft.fft2(np.fft.ifftshift(comp_object))      ))           ###    fft + ifftshift 
# plt.imshow ( np.angle(    np.fft.fft2(np.fft.ifftshift(comp_object))[0,:,:] ))     ###      fft ifftshift






# plt.imshow (   abs(   np.fft.fftshift(comp_object)       ))                      ## fft to object only    
# plt.imshow (   np.angle(      np.fft.fftshift(comp_object)            ))          ## fft to object only

# plt.imshow ( abs(    np.fft.fftshift(   np.fft.fft2(comp_object[0,:,:])  ) ))              ### fft shift + fft
# plt.imshow ( np.angle(       np.fft.fftshift(   np.fft.fft2(comp_object[0,:,:])  ) ))     ##   fft shift + fft


# plt.imshow ( abs(    np.fft.fftshift(   np.fft.fft2( np.fft.ifftshift(comp_object)) )[0,:,:])  )           ###  fftshift + fft + ifft
# plt.imshow ( np.angle(       np.fft.fftshift(   np.fft.fft2(np.fft.ifftshift(comp_object)  ) )[0,:,:]             ))   ###  fftshift + fft + ifft


# plt.imshow(abs(np.fft.fftshift(comp_object[0,:,:])))

# FT_npy = np.fft.fftshift(   np.fft.fft2( np.fft.ifftshift(comp_object)) )


# plt.imshow ( abs(    np.fft.fftshift(   np.fft.ifft2( np.fft.ifftshift(FT_npy)) )[0,:,:])  )           ###  fftshift + fft + iffts

# plt.imshow ( np.angle(    np.fft.fftshift(   np.fft.ifft2( np.fft.ifftshift(FT_npy)) )[0,:,:])  )           ###  fftshift + fft + ifft


# A = A.permute()
# ########################################


# AT = torch.tensor(comp_object)

# AT.shape
# AT_R = torch.view_as_real(AT)

# AT_R.shape 

    
# plt.imshow ( abs(       torch.view_as_complex(fftshift(torch.fft( ifftshift(AT_R),2)))[0,:,:])     )          ###  fftshift + fft + ifft
# plt.imshow( torch.angle  (         fftshift(torch.view_as_complex(torch.fft( torch.view_as_real(ifftshift(AT_R)),2)))[0,:,:] ))



# FT_torch = torch.view_as_real( fftshift(torch.view_as_complex(torch.fft( torch.view_as_real(ifftshift(torch.view_as_complex(AT_R))),2))))

# FT_torch.dtype
# FT_torch.shape

# plt.imshow ( abs(       fftshift(torch.view_as_complex(torch.ifft( torch.view_as_real(ifftshift(torch.view_as_complex(FT_torch))),2)))[0,:,:]     ))          ###  fftshift + fft + ifft
# plt.imshow( torch.angle  (         fftshift(torch.view_as_complex(torch.ifft( torch.view_as_real(ifftshift(torch.view_as_complex(FT_torch))),2)))[0,:,:] ))


# A.shape

# BB = A[0,:,:,0]
# BBB = A[0,:,:,1]



# BBBB = np.multiply(  BB,  np.exp(1j*np.pi*(  BBB))) 

# plt.imshow(BB)
# plt.imshow(BBB)

# BBBB.shape

# BBBB=torch.tensor(BBBB)


# BFT1 = abs(torch.view_as_complex(fftshift(torch.fft(torch.view_as_real(   ifftshift(BBBB)  ) ,2))))

# BFT2 = abs(torch.view_as_complex(fftshift(torch.fft(torch.view_as_real(   (BBBB)  ) ,2))))


# plt.imshow( BFT1  )
# plt.imshow( BFT2  )

# plt.imshow(abs(BFT1 - BFT2))
# plt.imshow(abs( fftshift(torch.view_as_complex(torch.ifft(ifftshift(BFT), 2)))))

# plt.imshow(torch.angle( torch.view_as_complex(torch.ifft(ifftshift(BFT), 2))))

