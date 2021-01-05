
import torch.nn.functional as F
import torch.nn as nn
from torch import ones , ones_like, absolute, fft ,ifft, sqrt, mul,div, stack, tensor, view_as_complex, view_as_real, angle, cos, sin, mean, clamp
from utils import fftshift ,ifftshift, Settings , gseq ,   indices , CTF_s, n_iter ,Phase_shift
from model_parts import *
import matplotlib.pyplot as plt
import torch as torch
from config import config_func
opt = config_func()

import numpy as np 







class UNet_SBS(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_SBS, self).__init__()
        
        
        self.input_c = n_channels
        self.n_channels = n_channels * 4
        self.n_classes = n_classes 

        self.conv1 = DoubleConv(self.input_c, 2*self.n_channels)
        self.conv2 = DoubleConv(2 * self.n_channels, 4*self.n_channels)
        self.down1 = Down( 4*self.n_channels,  8* self.n_channels)
        self.down2 = Down( 8*self.n_channels,  16* self.n_channels)
        self.conv3=  DoubleConv(16*self.n_channels,  8* self.n_channels)
        
        
        self.up1 = Up(8*self.n_channels, 4*self.n_channels)
        self.up2 = Up(4*self.n_channels, 2*self.n_channels)
        
        
        self.outc = OutConv(2*self.n_channels, n_classes)

    def forward(self, x):
        
        x0 = self.conv1(x)
        
        x1 = self.conv2(x0)
        
        x2 = self.down1(x1)
        
        x3 = self.down2(x2)
        
        x4 = self.conv3(x3)
        
        x5 = self.up1(x4, x2)
        
        x6 = self.up2(x5, x1)
        
        x9 = self.outc(x6)
        

        return x9




class UNet(nn.Module):
    def __init__(self, n_channel, n_classes, bilinear=True):
        super(UNet, self).__init__()
        
        self.n_channels = n_channel * 2 
        self.n_classes = n_classes 

        self.conv1 = DoubleConv(n_channel, 2*self.n_channels)
        self.conv2 = DoubleConv(2 * self.n_channels, 4*self.n_channels)
        self.down1 = Down( 4*self.n_channels,  8* self.n_channels)
        self.down2 = Down( 8*self.n_channels,  16* self.n_channels)
        self.conv3=  DoubleConv(16*self.n_channels,  8* self.n_channels)
        
        
        self.up1 = Up(8*self.n_channels, 4*self.n_channels)
        self.up2 = Up(4*self.n_channels, 2*self.n_channels)
        
        self.up3 = Up_nocat(2*self.n_channels, 2*self.n_channels)
        self.up4 = Up_nocat(2 *self.n_channels, self.n_channels)
        

        self.outc = OutConv(self.n_channels, n_classes)

    def forward(self, x):
        
        x0 = self.conv1(x)
        
        x1 = self.conv2(x0)
        
        x2 = self.down1(x1)
        
        x3 = self.down2(x2)
        
        x4 = self.conv3(x3)
        
        x5 = self.up1(x4, x2)
        
        x6 = self.up2(x5, x1)
        
        x7 = self.up3(x6)
        
        x8 = self.up4(x7)
        
        x9 = self.outc(x8)
        

        return x9
    
    
    
class Generator_go(nn.Module):
    def __init__(self, n_channel, n_classes):
        super(Generator_go, self).__init__()
        
        self.n_channels = n_channel 
        self.n_classes = n_classes 

        self.CC = 32


        # self.flatten = Flatten()
        # self.dense1 = Dense(2 * self.n_channels * 128 * 128 ,1024)
        # self.dense1 = Dense(2 * self.n_channels * 128 * 128 ,1024)

        self.flatten_a = Flatten(2)
        self.flatten_p = Flatten(2)
        
        self.dense1_a =  Dense( 32*32  , 128*128)
        self.dense1_p =  Dense( 32*32  , 128*128)

        # self.dense2_a =  Dense( 32*32 , 128*128)
        # self.dense2_p =  Dense( 32*32 , 128*128)

        self.reshape_a = Reshape(-1, self.n_channels,  128 ,  128 )
        self.reshape_p = Reshape(-1, self.n_channels,  128 ,  128 )
        
                
        # self.dua = nn.ConvTranspose2d(self.n_channels , 1 , 4,stride =4 ,padding = 0 )
        # self.dup = nn.ConvTranspose2d(self.n_channels , 1 , 4,stride =4 ,padding = 0 )

        self.dua = DoubleConv(self.n_channels , self.n_channels  )
        self.dup = DoubleConv(self.n_channels , self.n_channels  )

        
        self.rconvout_a = OutConv (self.n_channels ,1 )
        self.rconvout_p = OutConv (self.n_channels ,1 )
        
        
        # self.rout_a = OutConv (1 ,1 )
        # self.rout_p = OutConv (1 ,1 )
        
        
        
        ########################################################################
        
        s = 2
        p = 1
        k = 4
        

        self.conv0_a = DoubleConv_residual (self.n_channels , self.CC)
        self.conv0_p = DoubleConv_residual (self.n_channels , self.CC)

        self.conv1_a = DoubleConv_residual ( self.CC , self.CC )
        self.conv1_p = DoubleConv_residual ( self.CC , self.CC )


        self.down1_a = nn.MaxPool2d(2)
        self.down1_p = nn.MaxPool2d(2)


        self.conv2_a = DoubleConv_residual ( self.CC,  2*self.CC)
        self.conv2_p = DoubleConv_residual ( self.CC,  2*self.CC)


        self.down2_a = nn.MaxPool2d(2)
        self.down2_p = nn.MaxPool2d(2)


        self.conv3_a = DoubleConv_residual ( 2*self.CC,  4*self.CC)
        self.conv3_p = DoubleConv_residual ( 2*self.CC,  4*self.CC)

    
        self.up3_a = nn.ConvTranspose2d(4*self.CC , 4*self.CC , k,stride =2 ,padding = p )
        self.up3_p = nn.ConvTranspose2d(4*self.CC , 4*self.CC , k,stride =2 ,padding = p )


        self.conv4_a = DoubleConv_residual ( 6*self.CC,  3*self.CC)
        self.conv4_p = DoubleConv_residual ( 6*self.CC,  3*self.CC)

    
        self.up4_a = nn.ConvTranspose2d(3*self.CC , 3*self.CC , k,stride =2 ,padding = p )
        self.up4_p = nn.ConvTranspose2d(3*self.CC , 3*self.CC , k,stride =2 ,padding = p )


        self.conv5_a = DoubleConv_residual ( 4*self.CC,  2*self.CC)
        self.conv5_p = DoubleConv_residual ( 4*self.CC,  2*self.CC)

    
        self.up5_a = nn.ConvTranspose2d(2*self.CC , 2*self.CC , k,stride =2 ,padding = p )
        self.up5_p = nn.ConvTranspose2d(2*self.CC , 2*self.CC , k,stride =2 ,padding = p )


        self.conv6_a = DoubleConv_residual ( 2*self.CC,  2*self.CC)
        self.conv6_p = DoubleConv_residual ( 2*self.CC,  2*self.CC)


        self.up6_a = nn.ConvTranspose2d(2*self.CC , 2*self.CC , k,stride =2 ,padding = p )
        self.up6_p = nn.ConvTranspose2d(2*self.CC , 2*self.CC , k,stride =2 ,padding = p )


        self.conv7_a = DoubleConv_residual ( 2*self.CC,  1*self.CC)
        self.conv7_p = DoubleConv_residual ( 2*self.CC,  1*self.CC)
        
        
        
        self.convout_a = OutConv (1*self.CC ,1 )
        self.convout_p = OutConv (1*self.CC ,1 )
        
        self.out_a = OutConv (2 ,1 )
        self.out_p = OutConv (2 ,1 )




    def forward(self, x):
         
        
        xflta = self.flatten_a(x)
        xfltp = self.flatten_p(x)
        
        
        
        xd1a = self.dense1_a(xflta)
        xd1p = self.dense1_p(xfltp)
        
        # print(xd1a.shape, 'xd1a')
        
        # xd2a = self.dense2_a(xd1a)
        # xd2p = self.dense2_p(xd1p)
        
        # print(xd2a.shape, 'xd2a')
        
        xra = self.reshape_a ( xd1a )
        xrp = self.reshape_p ( xd1p )
        
        # print(xra.shape, 'xra')
        
        xddda = self.dua(xra)
        xdddp = self.dup(xrp)
        
        
        xrao = self.rconvout_a(xddda)
        xrpo = self.rconvout_p(xdddp)
        
        # xraoo = self.rout_a(xrao)
        # xrpoo = self.rout_p(xrpo)
        
        
        ##############################################
        
        x0a = self.conv0_a(x)
        x0p = self.conv0_p(x)
        
        x1a = self.conv1_a (x0a)
        x1p = self.conv1_p (x0p)
        
        x1da= self.down1_a(x1a)
        x1dp= self.down1_p(x1p)
        
        
        x2a = self.conv2_a(x1da)
        x2p = self.conv2_p(x1dp)

        x2da = self.down2_a(x2a)
        x2dp = self.down2_p(x2p)

        x3a = self.conv3_a(x2da)
        x3p = self.conv3_p(x2dp)
        
        x3ua = self.up3_a(x3a )
        x3up = self.up3_p(x3p )
        
        
        x3ca = torch.cat([x2a , x3ua] , dim=1)
        x3cp = torch.cat([x2p , x3up] , dim=1)
        
        
        x4a = self.conv4_a(x3ca)
        x4p = self.conv4_p(x3cp)

        x4ua = self.up4_a(x4a)
        x4up = self.up4_p(x4p)

        
        x4ca = torch.cat( [ x1a , x4ua] , dim=1)
        x4cp = torch.cat( [ x1p , x4up] , dim=1)
        
        
        x5a = self.conv5_a(x4ca)
        x5p = self.conv5_p(x4cp)


        x5ua = self.up5_a(x5a)
        x5up = self.up5_p(x5p)

        x6a = self.conv6_a(x5ua)
        x6p = self.conv6_p(x5up)

        x6ua = self.up6_a(x6a)
        x6up = self.up6_p(x6p)

        x7a = self.conv7_a(x6ua)
        x7p = self.conv7_p(x6up)

        
        x8a = self.convout_a(x7a)
        x8p = self.convout_p(x7p)
        
        
        xfa = torch.cat([xrao, x8a] , dim=1)
        xfp = torch.cat([xrpo, x8p] , dim=1)
        
        
        xoa = self.out_a(xfa)
        xop = self.out_p(xfp)
        
        
        # xoa = xrao + x8a
        # xop = xrpo + x8p
        
        x_out = torch.cat([xoa , xop] , dim=1)
        
        
        
        
        return x_out
    
    
    
    



class Discriminator_go(nn.Module):
    def __init__(self, n_channel, n_classes, bilinear=True):
        super(Discriminator_go, self).__init__()
        
        
        self.n_channel = n_channel
        self.n_channels = 2* self.n_channel 
        self.n_classes = n_classes 

        self.conv1 = DoubleConv_residual(self.n_channel, 2*self.n_channels)                     ## 128 128
        self.conv1_1 = DoubleConv_residual(2*self.n_channels, 2*self.n_channels)                     ## 128 128
        self.conv1_2 = DoubleConv_residual(2*self.n_channels, 2*self.n_channels)                     ## 128 128
                            ## 128 128
        # self.conv1_3 = DoubleConv_residual(2*self.n_channels, 4*self.n_channels)                     ## 128 128
        # self.conv1_4 = DoubleConv_residual(4*self.n_channels, 4*self.n_channels)                     ## 128 128
        self.down1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv_residual( 2*self.n_channels , 4*self.n_channels)           ## 64 64 
        self.conv2_1 = DoubleConv_residual( 4*self.n_channels , 4*self.n_channels)           ## 64 64 
        # self.down2 = nn.MaxPool2d(2)
        # self.conv3 = DoubleConv_residual( 4*self.n_channels , 8*self.n_channels)           ## 64 64 
        # self.conv3_1 = DoubleConv_residual( 8*self.n_channels , 8*self.n_channels)           ## 64 64 
        self.final = nn.Sequential(
            nn.Conv2d( 4*self.n_channels ,1,64,1,0,  bias='false')           ## 64 64 
            ,nn.Sigmoid()
            )
        # self.down3 = nn.MaxPool2d(2)


        # self.conv4 = DoubleConv_residual( 8*self.n_channels , 16*self.n_channels)             ## 16 16 
        # self.ups = nn.Sequential(
            
        # nn.Upsample(scale_factor=2 , mode = 'nearest'),
        # nn.Upsample(scale_factor=2 , mode = 'nearest')
            
        #     )
        
        
        #################
        
        # 
        # self.flatten = Flatten()
        # self.dense1 = Dense(8 * self.n_channels * 32 * 32 ,n_classes)
        # self.dense2 = Dense(2 , 2 )
        # self.dense3 = Dense_out(2 , n_classes )

    def forward(self, x0):
        
        # usmp = self.ups(smp)
        # x=  torch.cat([usmp, x0] , axis=1)
        
        x1 = self.conv1(x0)
        x1_1 = self.conv1_1(x1)
        x1_2 = self.conv1_2(x1_1)
        # x1_3 = self.conv1_3(x1_2)
        # x1_4 = self.conv1_4(x1_3)
        
        
        x2 = self.down1(x1_2)
        
        x3 = self.conv2(x2)
        x33 = self.conv2_1(x3)
        # x4 = self.down2(x33)
        # x5 = self.conv3(x4)
        # x55 = self.conv3_1(x5)
        xf = self.final(x33)

        
        # x7 = self.conv4(x6)
        
        # print(x6.shape , 'x6_shape')
        # x8 = self.flatten(x55)
        # print(x8.shape , 'x8_shape')
        
        # print( 8 * self.n_channels * 16 * 16 , 'hoho')

        # x9 = self.dense1(x8)
        
        # x10 = self.dense2(x9)
    
        # x_out = self.dense3(x10)
        # print(x_out.shape , 'x_out.shape')
        

        return xf
    
    



class Discriminator_low(nn.Module):
    def __init__(self, n_channel, n_classes):
        super(Discriminator_low, self).__init__()
        
        self.n_channels = 121
        self.n_classes = n_classes 

        self.conv1 = DoubleConv_residual(self.n_channels, self.n_channels)                     ## 128 128
                            ## 128 128
        
        self.conv2 = DoubleConv_residual( self.n_channels , 2*self.n_channels)           ## 64 64 
        
        
        
                
        self.down1 = nn.MaxPool2d(2)
        
        self.conv3 = DoubleConv_residual( 2*self.n_channels , 2*self.n_channels)           ## 64 64 
        self.conv4 = DoubleConv_residual( 2*self.n_channels , 2*self.n_channels)           ## 64 64 
        #################
        
        self.flatten = Flatten()
        self.dense1 = Dense_out(2 * self.n_channels * 16 * 16 ,n_classes )
        # self.dense2 = Dense(256 , 256 )
        # self.dense3 = Dense_out(256 , n_classes )

    def forward(self, x0):
        
        
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        
        
        x3 = self.down1(x2)
        
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x8 = self.flatten(x5)        
        x9 = self.dense1(x8)        
        # x10 = self.dense2(x9)    
        # x_out = self.dense3(x10)
        # print(x_out.shape , 'x_out.shape')
        

        return x9
    
    
    





    
        
class DenseKnet(nn.Module):
    def __init__(self, n_size, n_channel):
        
        super(DenseKnet, self).__init__()
        self.uprate = 4
        self.n_size = n_size 
        self.n_channel = n_channel 
        


        self.flatten1 = Flatten()                                 ## BS x  25 x  ( 16 x 16   ) 
        self.dense1_comp = Dense(  self.n_channel *  (self.n_size)**2  ,  ((self.uprate *self.n_size)**2)    )
        # self.dense2_comp = Dense( 100  ,   ((self.uprate *self.n_size)**2)   )
        
        
        self.reshape1_tocomp = Reshape(-1, 1, (self.uprate *self.n_size) ,  ((self.uprate *self.n_size) ))
        
        #self.convocp = OutConv(50, 2)
        
       #  self.flatten3 = Flatten()                                 ## BS x  25 x  ( 16 x 16   ) 
       # # self.dense3_comp = Dense(  ((self.uprate * self.n_size)**2)  , ((self.uprate *self.n_size)**2)   )
       #  self.reshape3_tocomp = Reshape(-1,  1, (self.uprate *self.n_size) , (self.uprate *self.n_size) )
        
       #  #self.transposed1 =  Transpose_conv(self.n_channel  , self.n_channel)
       #  #self.transposed2 =  Transpose_conv(2*self.n_channel  , self.n_channel)
       #  self.convt_1 = DoubleConv(1, self.n_channel)
        
        self.convt_2 = DoubleConv(1, self.n_channel)
        self.convt_3 = DoubleConv(self.n_channel, self.n_channel)
        self.convt_4 = DoubleConv(self.n_channel, self.n_channel)
        self.convt_5 = OutConv( self.n_channel, 1)
        
    def forward(self, x):
        
        #
        # print(x.shape)
        x1 = self.flatten1(x)
        # print(x1.shape)
        x1 = self.dense1_comp(x1)
        # x1 = self.dense2_comp(x1)
        x1 = self.reshape1_tocomp(x1)
        x1 = self.convt_2(x1)
        x1 = self.convt_3(x1)
        x1 = self.convt_4(x1)
        x1 = self.convt_5(x1)
        
        
       
        return x1
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    