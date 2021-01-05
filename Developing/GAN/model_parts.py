import torch
import torch.nn as nn
import torch.nn.functional as F


def GenConvBlock(n_conv_layers, in_chan, out_chan, feature_maps):
    conv_block = [nn.Conv2d(in_chan, feature_maps, 3, 1, 1),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    for _ in range(n_conv_layers - 2):
        conv_block += [nn.Conv2d(feature_maps, feature_maps, 3, 1, 1),
                       nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    return nn.Sequential(*conv_block, nn.Conv2d(feature_maps, out_chan, 3, 1, 1))


class GenResConvBlock(nn.Module):
    def __init__(self, n_conv_layers, in_chan, out_chan, feature_maps):
        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.feature_maps = feature_maps
        
        self.conv_block = [nn.Conv2d(in_chan, feature_maps, 3, 1, 1),
                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]    
        for _ in range(n_conv_layers - 2):
            self.conv_block += [nn.Conv2d(feature_maps, feature_maps, 3, 1, 1),
                           nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        self.conv_block = nn.Sequential(*self.conv_block, nn.Conv2d(feature_maps, out_chan, 3, 1, 1))
        self.conv_block_F = nn.Sequential(nn.Conv2d(feature_maps , out_chan, 3, 1, 1))
        
        
        
    def forward (self, x):
        self.shortcut = x
        self.conv_out = self.conv_block(x)
        # return self.conv_out
        return self.shortcut + self.conv_out
        
        








def Gen(n_conv_layers, in_chan, out_chan, feature_maps):
    conv_block = [nn.Conv2d(in_chan, feature_maps, 3, 1, 1),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    for _ in range(n_conv_layers - 2):
        conv_block += [nn.Conv2d(feature_maps, feature_maps, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    return nn.Sequential(*conv_block, nn.Conv2d(feature_maps, out_chan, 3, 1, 1))













class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)





class DoubleConv_residual(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
            
            
            
        self.Branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            
        )
        # self.activation_last = nn.ReLU(inplace=True)


    def forward(self, x):
        self.out1 = self.conv_first(x)
        self.out2=  self.Branch(self.out1)
        
        # self.output = self.activation_last( self.out1 + self.out2)
        self.output = self.out1 + self.out2
        
        return self.output





class DoubleConv_5(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

    
    
class Transpose_conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        #if not mid_channels:
        #    mid_channels = out_channels
        self.transposeconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4 ,stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),

        )
    def forward(self, x):
        return self.transposeconv(x)
    
    
    
    
class DoubleConv_Tanh(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv_tanh = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Tanh(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Tanh()
        )

    def forward(self, x):
        return self.double_conv_tanh(x)

class DoubleConv_Tanh_5(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv_tanh = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.Tanh(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.Tanh()
        )

    def forward(self, x):
        return self.double_conv_tanh(x)
    
    
    
class DoubleConv_Tanh_7(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv_tanh = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.Tanh(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.Tanh()
        )

    def forward(self, x):
        return self.double_conv_tanh(x)
    
    
    
    
    
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down , self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)







class Up1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up1 , self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)




class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up , self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv(in_channels*(2), out_channels)
        
        
    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        
        
        x = torch.cat([x2, x1], dim=1)
        
        x = self.conv(x)
        
        
        return x




class Up_nocat(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up_nocat , self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
        

    def forward(self, x1):
        x1 = self.up(x1)
        x = self.conv(x1)
        
        
        return x





class OutConv_relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
            )


    def forward(self, x):
        return self.conv(x)






class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            # nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
            )


    def forward(self, x):
        return self.conv(x)
    
    
    
    
    
    
class OutConv_Tanh(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_Tanh, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
        
    
class Dense(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dense, self).__init__()
        
        self.dense_layer = nn.Sequential(
            
            # nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels ,out_channels ),
            nn.LeakyReLU(0.2, inplace= True)
                        
            )
        
    def forward(self, x):
        return self.dense_layer(x)
    
    
class Dense_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dense_out, self).__init__()
        
        self.dense_layer = nn.Sequential(
            nn.Linear(in_channels ,out_channels ),
            nn.Sigmoid()
                        
            )
        
    def forward(self, x):
        return self.dense_layer(x)
    
        
    
class Flatten(nn.Module):
    def __init__(self, start_dim  = 1 , end_dim= -1):
        super(Flatten, self).__init__()
        self.flatten = nn.Flatten(start_dim, end_dim= -1)

    def forward(self, x):
        return self.flatten(x)
    
    
    
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        # print(self.shape , 'self.shape')
        return x.view(self.shape)