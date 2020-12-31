import numpy as np
import scipy.io as sio

from config import GenConfig
from FPM_utils import extract_indices_CTF, extract_FPM_images

from os import makedirs

from glob import glob

opt = GenConfig()

dataset_name = 'DIV2K_800_r1024_d128'

CTF, indices_e, indices_f, seq = extract_indices_CTF(opt)

load_dir = sorted(glob('../../../Data/%s/Valid/*.mat' % (dataset_name)))
save_dir = '../../../Data/AP/%s/array%d_iters%d/Valid' % (dataset_name, opt.array_size, opt.n_iters)

makedirs(save_dir, exist_ok = True)

for k in range(len(load_dir)):
    print('%d/%d' % (k, len(load_dir)))
    
    img_obj = sio.loadmat(load_dir[k])['img'] # (128, 128) (dtype: complex)
    img_FPM, img_FPM_ri, img_FPM_sqrt, img_FPM_comp = extract_FPM_images(img_obj, opt, indices_f, CTF) # (25, 32, 32) (dtype: real)
    
    objectRecover = np.ones((128, 128));
    objectRecoverFT = np.fft.fftshift(np.fft.fft2(objectRecover))
        
    for tt in range (0, opt.n_iters):
        for i in range(0, opt.array_size**2):
            kyl=indices_f["kyl"][i]; kyh=indices_f["kyh"][i]; kxl=indices_f["kxl"][i]; kxh=indices_f["kxh"][i]
            
            lowResFT = objectRecoverFT[kyl:kyh+1, kxl:kxh+1] * CTF
            
            im_lowRes = np.fft.ifft2(np.fft.ifftshift(lowResFT))
    
            im_lowRes = img_FPM_sqrt[i,:,:] * (im_lowRes) / abs(im_lowRes)
            
            lowResFT=np.fft.fftshift(np.fft.fft2(im_lowRes)) * CTF
            
            objectRecoverFT[kyl:kyh+1, kxl:kxh+1] = np.multiply((1-CTF), objectRecoverFT[kyl:kyh+1, kxl:kxh+1]) + lowResFT
    
    objectRecover = np.fft.ifft2(np.fft.ifftshift(objectRecoverFT))
    
    sio.savemat('%s/%s' % (save_dir, load_dir[k][-12:]), mdict={'img_AP':objectRecover})