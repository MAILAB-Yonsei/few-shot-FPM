import torch
import torch.nn as nn
from torch import sqrt, fft, ifft, stack, ones_like, mean, cat
from torch import unsqueeze, clamp, acos

def GenConvBlock(n_conv_layers, in_chan, out_chan, feature_maps):
    conv_block = [nn.Conv2d(in_chan, feature_maps, 3, 1, 1),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    for _ in range(n_conv_layers - 2):
        conv_block += [nn.Conv2d(feature_maps, feature_maps, 3, 1, 1),
                       nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    return nn.Sequential(*conv_block, nn.Conv2d(feature_maps, out_chan, 3, 1, 1))

def GenFcBlock(feat_list=[512, 1024, 1024, 512]):
    FC_blocks = []
    len_f = len(feat_list)
    for i in range(len_f - 2):
        FC_blocks += [nn.Linear(feat_list[i], feat_list[i+1]),
                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        
    return nn.Sequential(*FC_blocks, nn.Linear(feat_list[len_f - 2], feat_list[len_f - 1]))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.1, nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

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

def fft2(x):
    return fftshift(fft(x.permute(0,2,3,1), 2), [-2, -3]).permute(0,3,1,2)

def ifft2(x):
    return ifft(ifftshift(x.permute(0,2,3,1), [-2, -3]), 2).permute(0,3,1,2)

def AP_module(img_FPM, objectRecoverFT, opt, indices, CTF_stack):
    for k in range(opt.n_iters):
        for i in range(opt.array_size ** 2): # extraction
            FT_FPM_recon  = objectRecoverFT[:,:,indices[0][i]:indices[1][i]+1,indices[2][i]:indices[3][i]+1] * CTF_stack # (n_batch, 2, 32, 32)
            img_FPM_recon = ifft2(FT_FPM_recon)
            
            img_FPM_rur = img_FPM_recon[:,0,:,:] / absolute(img_FPM_recon)
            img_FPM_rui = img_FPM_recon[:,1,:,:] / absolute(img_FPM_recon)
            
            im_lowRes = stack([img_FPM[:,i,:,:] * img_FPM_rur, img_FPM[:,i,:,:] * img_FPM_rui], dim=1) # (n_batch, 2, 32, 32)
            lowResFT = fft2(im_lowRes) * CTF_stack # (n_batch, 32, 32, 2)
            
            objectRecoverFT[:,:,indices[0][i]:indices[1][i]+1, indices[2][i]:indices[3][i]+1] = (ones_like(CTF_stack) - CTF_stack) * objectRecoverFT[:,:,indices[0][i]:indices[1][i]+1, indices[2][i]:indices[3][i]+1] + lowResFT # (n_batch, 32, 32, 2)
    
    return objectRecoverFT

def phase_loss(out_uv, label_uv):    
    img_cos = torch.sum(out_uv * label_uv, 1)
    loss = mean(acos(clamp(img_cos, -1+1e-7, 1-1e-7)) ** 2)
    
    # print((acos(clamp(img_cos, -1+1e-7, 1-1e-7)) ** 2).min())
    # print((acos(clamp(img_cos, -1+1e-7, 1-1e-7)) ** 2).max())
    
    return loss

def absolute(x):
    return clamp(sqrt(x[:,0,:,:] ** 2 + x[:,1,:,:] ** 2), 1e-5)

def uau(x):
    x_mag = clamp(sqrt(x[:,0] ** 2 + x[:,1] ** 2), 1e-5) 
    x_unvr = x[:,0] / x_mag
    x_unvi = x[:,1] / x_mag
    
    x_unv = stack([x_unvr, x_unvi], dim=1)
    
    return x_unv

def split_munv(x):
    x_mag = absolute(x)
    # print('min of x_mag: %f' % x_mag.min())
    x_unvr = x[:,0,:,:] / x_mag
    x_unvi = x[:,1,:,:] / x_mag
    
    x_unv  = stack([x_unvr, x_unvi], dim = 1)
    
    return x_mag, x_unv

def norm_munv(x, concat):
    x_mag, x_unv = split_munv(x)
    
    x_mag_avg = mean(x_mag, (-1, -2))
    x_unv_avg = mean(x_unv, (-1, -2))
    
    x_uau = uau(x_unv_avg)
    
    x_mag_norm = []
    x_unv_norm = []
    
    for i in range(x.shape[0]):
        x_mag_norm.append(x_mag[i,:,:] - x_mag_avg[i])
        
        x_unv_r =   x_unv[i,0,:,:] * x_uau[i,0] + x_unv[i,1,:,:] * x_uau[i,1] # (128, 128)
        x_unv_i = - x_unv[i,0,:,:] * x_uau[i,1] + x_unv[i,1,:,:] * x_uau[i,0] # (128, 128)
        
        x_unv_norm.append(stack([x_unv_r, x_unv_i], 0))
    
    x_mag_norm = stack(x_mag_norm, 0)
    x_unv_norm = stack(x_unv_norm, 0)
        
    avg = [x_mag_avg, x_unv_avg]
        
    if concat == True:
        return cat([unsqueeze(x_mag_norm, 1), x_unv_norm], dim=1), avg
    else:
        return x_mag_norm, x_unv_norm, avg
    
def cat_munv(mag, unv):
    return cat([unsqueeze(mag, 1), unv], dim=1)

def rescale_ri(x_mag_norm, x_unv_norm, avg):
    x_mag_avg = avg[0]
    x_unv_avg = avg[1]
    
    x_uau = uau(x_unv_avg)
    
    x_mag = []
    x_unv = []
    
    for i in range(x_mag_norm.shape[0]):
        x_mag.append(x_mag_norm[i,:,:] + x_mag_avg[i])
        
        x_unv_r = x_unv_norm[i,0,:,:] * x_uau[i,0] - x_unv_norm[i,1,:,:] * x_uau[i,1] # (128, 128)
        x_unv_i = x_unv_norm[i,0,:,:] * x_uau[i,1] + x_unv_norm[i,1,:,:] * x_uau[i,0] # (128, 128)
        
        x_unv.append(stack([x_unv_r, x_unv_i], 0))
        
    x_mag = stack(x_mag, 0)
    x_unv = stack(x_unv, 0)
    x_ri = stack([x_mag * x_unv[:,0,:,:], x_mag * x_unv[:,1,:,:]], dim=1)
    
    return x_ri