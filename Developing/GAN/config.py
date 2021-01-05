import argparse

def config_func():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--add_path_str', type=str, default='', help='string to be added to directory name')
    parser.add_argument('--model_name', type=str, default='', help='model name, choose Unet_scale1 or 2 or 4')
    
    parser.add_argument('--dataset_name', type=str, default='DIV2K_800_resample128', help='name of the dataset')
    parser.add_argument('--acc_rate', type=float, default=5, help='acceleration rate')
    parser.add_argument('--acs_num', type=int, default=16, help='the number of acs lines')
    
    
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    
    # model specifications
    
    
    # optimizer, training
    parser.add_argument('--batch_size', type=int, default=2, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--clip_value', type=float, default=1., help='adam: clipvalue')
    
    # image size
    parser.add_argument('--im_height', type=int, default=64, help='size of image height')
    parser.add_argument('--im_width', type=int, default=64, help='size of image width')
    parser.add_argument('--channels', type=int, default=25, help='number of image channels')
    
    # etc
    parser.add_argument('--gpu_alloc', type=int, default=0, help='GPU allocation')
    parser.add_argument('--sample_interval', type=int, default=300, help='batch interval between sampling of images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='epoch interval between model checkpoints')
   
    parser.add_argument('--mask_index_train', type=int, default=0, help='train mask index (-1 means random sampling, -2: bottom-half sampling)')
    parser.add_argument('--mask_index_test',  type=int, default=0, help='test mask index (-1 means random sampling, -2: bottom-half sampling)')
    # parser.add_argument('--multiple_masks', type=int, default=False, help='train mask number')
    
    
    
    opt = parser.parse_args()
    
    # if opt.pretrain == True:
    #     parser.add_argument('--model_structure_pre', type=str, default='DOTA')
    #     parser.add_argument('--add_path_str_pre', type=str, default='HCP_MGH_T1w_acc5_acs16_maskRand_fm_64')
    #     parser.add_argument('--epoch_retrain', type=int, default=200)
    #     opt = parser.parse_args()
    # if opt.multiple_masks == True:
    #     parser.add_argument('--mask_num', type=str, default=5)
    
    print(opt)
    return opt


