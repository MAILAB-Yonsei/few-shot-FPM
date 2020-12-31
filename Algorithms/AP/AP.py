import numpy as np
import scipy.io as sio

from config import GenConfig
from FPM_utils import extract_indices_CTF, extract_FPM_images
from glob import glob

opt = GenConfig()

CTF, indices_e, indices_f, seq = extract_indices_CTF(opt)


glob()

img_obj = sio.loadmat('../../../Data/DIV2K_800_r1024_d128/Valid/0336_001.mat')['img']

img_FPM, img_FPM_ri, img_FPM_sqrt = extract_FPM_images(img_obj, opt, indices_f, CTF) # (25, 32, 32) (dtype: real)

objectRecover = np.ones((128, 128));
objectRecoverFT = np.fft.fftshift(np.fft.fft2(objectRecover))

loop = 10
for tt in range (0, loop):
    for i in range(0, opt.array_size**2):
        kyl=indices_f["kyl"][i]; kyh=indices_f["kyh"][i]; kxl=indices_f["kxl"][i]; kxh=indices_f["kxh"][i]
        
#         lowResFT = ((m1/m)**2) * objectRecoverFT[kyl:kyh+1,kxl:kxh+1]* CTF
        lowResFT = objectRecoverFT[kyl:kyh+1, kxl:kxh+1] * CTF
        
        im_lowRes = np.fft.ifft2(np.fft.ifftshift(lowResFT))
#         im_lowRes = (m/m1)**2 * np.multiply(imSeqLowRes[:,:,i2],np.exp(1j*np.angle(im_lowRes)))
        im_lowRes = img_FPM_sqrt[i,:,:] * (im_lowRes) / abs(im_lowRes) # Phase만 update하여 곱하는 과정 (im_lowRes)/abs(im_lowRes) 가 phase image
        
        lowResFT=np.fft.fftshift(np.fft.fft2(im_lowRes)) * CTF
        
        objectRecoverFT[kyl:kyh+1, kxl:kxh+1] = np.multiply((1-CTF), objectRecoverFT[kyl:kyh+1, kxl:kxh+1]) + lowResFT

objectRecover=np.fft.ifft2(np.fft.ifftshift(objectRecoverFT))

img_AP = objectRecover