#%% import
import os
from math import sqrt

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from importlib import import_module

from datasets import ImageDataset

from config import GenConfig
from utils import GenAddPathStr, save_images
from FPM_utils import extract_indices_CTF
from layer_utils import weights_init_normal, norm_munv, phase_loss, rescale_ri

#%% settings for deep learning
opt = GenConfig()

add_path_str = GenAddPathStr(opt)
os.makedirs('SavedModels_%s/%s' % (opt.model_name, add_path_str), exist_ok=True)
os.makedirs('Validation_%s/%s'  % (opt.model_name, add_path_str), exist_ok=True)

GeneratorNet = getattr(import_module('models'), opt.model_name)
CTF, indices_e, indices_f, seq = extract_indices_CTF(opt)

# Initialize generator and criterion
generator = GeneratorNet(opt, indices_f)
criterion = torch.nn.MSELoss()

cuda = True if torch.cuda.is_available() else False
if cuda:
    generator = generator.cuda()
    criterion = criterion.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('SavedModels_%s/%s/generator_%d.pth' % (opt.model_name, add_path_str, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    

# Optimizers
optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=10**-7)

# Tensor allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

n_FPM = opt.array_size ** 2

if opt.input == 'm':
    input_str = 'img_FPM'
    in_ch = n_FPM
if opt.input == 'm_sqrt':
    input_str = 'img_FPM_sqrt'
    in_ch = n_FPM
elif opt.input == 'ri':
    input_str = 'img_FPM_ri'
    in_ch = n_FPM * 2

if opt.label == 'mp':
    label_str = 'img_obj_mp'
    out_ch = 2
if opt.label == 'ri':
    label_str = 'img_obj_ri'
    out_ch = 2
if opt.label == 'm':
    label_str = 'img_obj_m'
    out_ch = 1
if opt.label == 'p':
    label_str = 'img_obj_p'
    out_ch = 1

input_FPM_train = Tensor(opt.batch_size, in_ch, opt.size_LR, opt.size_LR)
input_CTF_train = Tensor(opt.batch_size, opt.size_LR, opt.size_LR)

input_FPM_valid = Tensor(1, in_ch, opt.size_LR, opt.size_LR)
input_CTF_valid = Tensor(1, opt.size_LR, opt.size_LR)

input_obj_train = Tensor(opt.batch_size, out_ch, opt.size_HR, opt.size_HR)
input_obj_valid = Tensor(1, out_ch, opt.size_HR, opt.size_HR)
#%% other settings
#if opt.data_augment: # Data augmentation
#    aug_seq = GenAugSeq()
#else:
aug_seq = None

# Data loader
dataloader_train = DataLoader(ImageDataset(opt, CTF, indices_f), batch_size=opt.batch_size, shuffle=True)
dataloader_valid = DataLoader(ImageDataset(opt, CTF, indices_f, is_valid=True, test_num=1), batch_size=opt.batch_size, shuffle=False)

#%% -------------------------- ###
###  Training with validation  ###
### -------------------------- ###

if __name__ == "__main__":
    for i, valid_data in enumerate(dataloader_valid):
        valid_FPM = Variable(input_FPM_valid.copy_(valid_data[input_str]))
        valid_CTF = Variable(input_CTF_valid.copy_(valid_data['CTF']))
        valid_obj = Variable(input_obj_valid.copy_(valid_data[label_str]))
                
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, train_data in enumerate(dataloader_train):
            # Training
            train_FPM = Variable(input_FPM_train.copy_(train_data[input_str]))
            train_CTF = Variable(input_CTF_train.copy_(train_data['CTF']))
            train_obj = Variable(input_obj_train.copy_(train_data[label_str]))
            
            # train_lbl = train_obj
            _, train_lbl, train_avg = norm_munv(train_obj, concat=False)
            
            
            optimizer.zero_grad()
            train_rec_mag, train_rec_unv = generator(train_FPM, train_CTF)
            
            loss_train = phase_loss(train_rec_unv, train_lbl)
            
            loss_train.backward()
            optimizer.step()
            
            # Validation
            # valid_lbl = valid_obj
            _, valid_lbl, valid_avg = norm_munv(valid_obj, concat=False)
            
            # print(valid_avg[0][0])
            
            # print('label:')
            # print(valid_avg[1])
        
            valid_rec_mag, valid_rec_unv = generator(valid_FPM, valid_CTF)
            
            # print(valid_rec_unv.shape)
            # view_data(valid_rec_unv, Tensor, 'rl')
            # print(absolute(valid_rec_unv).mean())
            
            loss_valid = phase_loss(valid_rec_unv, valid_lbl)
    
            valid_rec = rescale_ri(valid_rec_mag, valid_rec_unv, valid_avg)
            
            
            # Print status
            print("[Epoch %d/%d] [Batch %d/%d] [Valid loss: %.4f]" %
                  (epoch, opt.n_epochs, i, len(dataloader_train), sqrt(loss_valid.item())))
            
            # Save validation images
            batches_done = epoch * len(dataloader_train) + i
            if batches_done % opt.sample_interval == 0:
                val_data = [valid_rec, valid_obj]
                save_images(val_data, loss_valid, epoch, batches_done, add_path_str, Tensor, opt)
           
        # Save model checkpoints
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
                torch.save(generator.state_dict(), 'SavedModels_%s/%s/generator_%d.pth' % (opt.model_name, add_path_str, epoch + 1))