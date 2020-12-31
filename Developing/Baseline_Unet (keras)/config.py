import argparse

def config_func():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--add_path_str', type=str, default='', help='string to be added to directory name')
    # parser.add_argument('--model_name', type=str, default='unet_scale4', help='model name, choose Unet_scale1 or 2 or 4')
    
    parser.add_argument('--dataset_name', type=str, default='DIV2K_800_resample128', help='name of the dataset')
    
    parser.add_argument('--epoch', type=int, default=16, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    
    # optimizer, training
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--clip_value', type=float, default=1., help='adam: clipvalue')
    
    # image size
    parser.add_argument('--im_height', type=int, default=128, help='size of image height')
    parser.add_argument('--im_width', type=int, default=128, help='size of image width')
    parser.add_argument('--channels', type=int, default=25, help='number of image channels')
    
    # etc
    parser.add_argument('--gpu_alloc', type=int, default=0, help='GPU allocation')
    parser.add_argument('--sample_interval', type=int, default=15, help='batch interval between sampling of images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='epoch interval between model checkpoints')
   
    parser.add_argument('--mask_index_train', type=int, default=0, help='train mask index (-1 means random sampling, -2: bottom-half sampling)')
    parser.add_argument('--mask_index_test',  type=int, default=0, help='test mask index (-1 means random sampling, -2: bottom-half sampling)')
    
    opt = parser.parse_args()
    
    print(opt)
    return opt