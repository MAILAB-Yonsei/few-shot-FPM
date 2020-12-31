import torch.nn as nn

from torch import stack, ones, unsqueeze
from layer_utils import GenConvBlock, fft2, ifft2, AP_module, norm_munv
# from utils import view_data

class model_norm(nn.Module):
    def __init__(self, opt, indices_f):
        super(model_norm, self).__init__()
    
        conv_block_mag = GenConvBlock(opt.conv, 1, 1, opt.fm)
        conv_block_unv = GenConvBlock(opt.conv, 2, 2, opt.fm)
        
        self.conv_block_mag = conv_block_mag
        self.conv_block_unv = conv_block_unv
        self.opt = opt
        self.indices_f = indices_f

    def forward(self, img_FPM, CTF):
        kyl_f=self.indices_f["kyl"]; kyh_f=self.indices_f["kyh"]; kxl_f=self.indices_f["kxl"]; kxh_f=self.indices_f["kxh"]
        indices = [kyl_f, kyh_f, kxl_f, kxh_f]
        
        objectRecover = ones(img_FPM.shape[0], 2, self.opt.size_HR, self.opt.size_HR).cuda() # (n_batch, 128, 128, 2)
        objectRecoverFT = fft2(objectRecover)
        
        CTF_stack = stack([CTF, CTF], dim = 1) # (n_batch, 32, 32, 2)
        
        objectRecoverFT = AP_module(img_FPM, objectRecoverFT, self.opt, indices, CTF_stack)
        objectRecover   = ifft2(objectRecoverFT)
        
        OR_mag, OR_unv, _ = norm_munv(objectRecover, concat=False)

        OR_mag_rec = self.conv_block_mag(unsqueeze(OR_mag, 1))[:,0,:,:]
        # OR_unv_rec = OR_unv
        OR_unv_mag, OR_unv_rec, _ = norm_munv(self.conv_block_unv(OR_unv), concat=False)
    
        return OR_mag_rec, OR_unv_rec