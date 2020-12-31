import numpy as np
import math

import scipy.io as sio

#from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

from layer_utils import absolute

def GenAddPathStr(opt):
    add_path_str = 'array%d' % opt.array_size
    add_path_str += opt.add_path_str
    return add_path_str

#def GenAugSeq():
#    aug_seq = iaa.SomeOf((0, None), [iaa.Affine(rotate=90),
#                                     iaa.Flipud(1.0),
#                                     iaa.Fliplr(1.0),
#                                     iaa.PiecewiseAffine(scale=0.01)])
#    return aug_seq

def save_images(val_data, loss_valid, epoch, batches_done, add_path_str, Tensor, opt):
    titles = ['Reconstructed (epoch=%d, loss=%.4f)' % (epoch, math.sqrt(loss_valid)), 'Label']
    fig, axs = plt.subplots(2, 3)
    
    for col, image in enumerate([val_data[0], val_data[1]]):
        # Tensor to numpy, and extract magnitude
        img_np = Tensor.cpu(image.detach()).numpy()
    
        img_np1 = img_np[0,0,:,:]
        if opt.label == 'mp' or opt.label == 'ri':
            img_np2 = img_np[0,1,:,:]
                 
        if opt.label == 'mp':
            img_np = np.multiply(img_np1, np.exp(1j * img_np2))
        if opt.label == 'ri':
            img_np = img_np1 + 1j * img_np2
        if opt.label == 'm' or opt.label == 'p':
            img_np = img_np1
            
        img_abs    = np.abs(img_np)
        img_angle  = np.angle(img_np)
        kspace_np  = np.fft.fftshift(np.fft.fft2(img_np))
        kspace_abs = np.abs(kspace_np) ** 0.2
        
        titles_mag = 'min: %.4f, mean: %.4f, max: %.4f' % (img_abs.min(), img_abs.mean(), img_abs.max())
        titles_ang = 'min: %.4f, mean: %.4f, max: %.4f' % (img_angle.min(), img_angle.mean(), img_angle.max())
        
        axs[col, 0].imshow(kspace_abs, cmap='gray', vmin=0, vmax=10)
        axs[col, 0].set_title(titles[col], fontdict={'fontsize': 5})
        axs[col, 0].axis('off')
        axs[col, 1].imshow(img_abs, cmap='gray', vmin=0, vmax=1)
        axs[col, 1].set_title(titles_mag, fontdict={'fontsize': 5})
        axs[col, 1].axis('off')
        # axs[col, 2].imshow(img_angle, cmap='gray',vmin=-np.pi,vmax=np.pi)
        axs[col, 2].imshow(img_angle, cmap='gray',vmin=-np.pi,vmax=np.pi)
        axs[col, 2].set_title(titles_ang, fontdict={'fontsize': 5})
        axs[col, 2].axis('off')
        
        if col == 0:
            sio.savemat("Validation_%s/%s/%d_recon.mat" % (opt.model_name, add_path_str, batches_done), mdict={'img_abs':img_abs, 'img_angle':img_angle})
        elif col == 1:
            sio.savemat("Validation_%s/%s/%d_label.mat" % (opt.model_name, add_path_str, batches_done), mdict={'img_abs':img_abs, 'img_angle':img_angle})
        
    fig.savefig("Validation_%s/%s/%d.png" % (opt.model_name, add_path_str, batches_done), dpi = 300)
    
    
    plt.close()
    
def view_data(img, Tensor, dtype):
    if dtype == 'rl':
        img_mag  = absolute(img)[0,:,:]
        img_real = img[0,0,:,:]
        img_imag = img[0,1,:,:]
        
        img_mag  = Tensor.cpu(img_mag.detach()).numpy()
        img_real = Tensor.cpu(img_real.detach()).numpy()
        img_imag = Tensor.cpu(img_imag.detach()).numpy()
        
        plt.figure()
        plt.imshow(img_mag, vmin=0, vmax=1)
        plt.figure()
        plt.imshow(img_real, vmin=0, vmax=1)
        plt.figure()
        plt.imshow(img_imag, vmin=0, vmax=1)
        
    elif dtype == 'mp':
        img_mag = img[0,0,:,:]
        img_mag = Tensor.cpu(img_mag.detach()).numpy()
        
        plt.figure()
        plt.imshow(img_mag, vmin=0, vmax=1)     
    