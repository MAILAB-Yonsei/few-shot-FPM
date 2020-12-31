import argparse

def GenConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--add_path_str', type=str, default='', help='string to be added to directory name')
    parser.add_argument('--model_name', type=str, default='model3', help='model name')
    parser.add_argument('--dataset_name', type=str, default='DIV2K_800_r1024_d128', help='name of the dataset')
    parser.add_argument('--input', type=str, default='m_sqrt', help='input type') # mp, ri, m_sqrt p
    parser.add_argument('--label', type=str, default='ri', help='mp, ri, m, p') # mp, ri, m, p
    
    # FPM parameters
    parser.add_argument('--size_HR', type=int, default=128)    
    parser.add_argument('--size_LR', type=int, default=32)    
    parser.add_argument('--array_size',  type=int, default=9)
    parser.add_argument('--NA', type=float, default=0.12)       # 128: 0.15 / 64: 0.1
    parser.add_argument('--LEDgap', type=float, default=6)      # 128: 12   / 64: 7
    parser.add_argument('--LEDheight', type=float, default=50)  # 128: 50   / 64: 45
    parser.add_argument('--spsize', type=float, default=1.8e-6) # 128: 1.5  / 64: 2.4
    
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.00005, help='adam: learning rate') # start: 0.00005
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
        
    parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--sample_interval', type=int, default=10, help='interval between sampling of images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
    parser.add_argument('--data_augment', type=bool, default=False, help='32-fold data augmentation')
    
    # network parameters
    parser.add_argument('--n_iters', type=int, default=10)
    parser.add_argument('--conv', type=int, default=20)
    parser.add_argument('--in_ch', type=int, default=2)
    parser.add_argument('--out_ch', type=int, default=2)
    parser.add_argument('--fm', type=int, default=64)
    
    opt = parser.parse_args()
    
    print(opt)
    
    return opt