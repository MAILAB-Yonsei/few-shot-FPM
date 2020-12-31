from glob import glob
import scipy.io as sio
import numpy as np

from torch.utils.data import Dataset

from FPM_utils import extract_FPM_images

class ImageDataset(Dataset):
    def __init__(self, opt, CTF, indices_f, is_valid=False, is_testing=False, test_num=1, aug_seq=None):
        if is_testing:
            self.file_dir = sorted(glob('../../../Data/%s/Test/*.mat' % (opt.dataset_name)))
            if test_num != -1: # -1 means using all data for testing
                self.file_dir = self.file_dir[:test_num]
        elif is_valid:
            self.file_dir = sorted(glob('../../../Data/%s/Valid/*.mat' % (opt.dataset_name)))
            if test_num != -1: # -1 means using all data for testing
                self.file_dir = self.file_dir[:test_num]
        else:
            self.file_dir = sorted(glob('../../../Data/%s/Train/*.mat' % (opt.dataset_name)))

        self.is_testing = is_testing
        self.aug_seq = aug_seq
        self.opt = opt
        self.CTF = CTF
        self.indices_f = indices_f

    def __getitem__(self, index):
        img_obj = sio.loadmat(self.file_dir[index % len(self.file_dir)])['img'] # (128, 128) (dtype: complex)
        img_FPM, img_FPM_ri, img_FPM_sqrt, img_FPM_comp = extract_FPM_images(img_obj, self.opt, self.indices_f, self.CTF) # (25, 32, 32) (dtype: real)
        
        h, w = img_obj.shape
        img_obj_ri = np.reshape(img_obj, (h, w, 1))
        img_obj_ri = np.concatenate((np.real(img_obj_ri), np.imag(img_obj_ri)), axis = -1)
        
        img_obj_mp = np.reshape(img_obj, (h, w, 1))
        img_obj_m  = np.absolute(img_obj_mp)
        img_obj_p  = np.angle(img_obj_mp)
        
        img_obj_mp = np.concatenate((np.absolute(img_obj_mp), np.angle(img_obj_mp)), axis = -1)
    
        img_obj_ri = np.transpose(img_obj_ri, (2,0,1))
        img_obj_mp = np.transpose(img_obj_mp, (2,0,1))
        img_obj_m  = np.transpose(img_obj_m, (2,0,1))
        img_obj_p  = np.transpose(img_obj_p, (2,0,1))

        return {'img_FPM': img_FPM, 'img_FPM_ri': img_FPM_ri, 'img_FPM_sqrt': img_FPM_sqrt,
                'img_obj_ri': img_obj_ri, 'img_obj_mp': img_obj_mp, 'img_obj_m': img_obj_m, 'img_obj_p': img_obj_p, 
                'CTF': self.CTF, 'indices_f': self.indices_f}

    def __len__(self):
        return len(self.file_dir)