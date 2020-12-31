from __future__ import print_function
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, BatchNormalization, concatenate
#from group_normalization import GroupNormalization

def unet(sx=32, sy=32, nch=81):

    nfilter = nch
    inputs = Input(shape=(sx, sy, nch))                                                                            # 512*512*1
    
    conv1 = Conv2D(nfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)    # 512*512*8
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(nfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)     # 512*512*8
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 80, 80, 64, 16                                                 # 256*256*16
    
    conv2 = Conv2D(nfilter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)   # 256*256*16
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(nfilter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)   # 256*256*16
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 40, 40, 32, 32                                                 # 128*128*16
    
    conv3 = Conv2D(nfilter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)   # 128*128*32
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(nfilter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)   # 128*128*32
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 20, 20, 16, 64                                                 #  64*64*32
    
    conv4 = Conv2D(nfilter*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)   # 64*64*64
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(nfilter*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)   # 64*64*64
    conv4 = BatchNormalization()(conv4)

    
    up6 = Conv2D(nfilter*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4)) # 64*64*128
    merge6 = concatenate([conv3,up6], axis = 3) 
    print("merge6 shape : ", merge6.shape)
    conv6 = Conv2D(nfilter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(nfilter*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6) # 20, 20, 16, 128
    conv6 = BatchNormalization()(conv6)
    
    up7 = Conv2D(nfilter*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv2,up7], axis = 3)
    print("merge7 shape : ", merge7.shape)
    conv7 = Conv2D(nfilter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(nfilter*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7) # 40, 40, 32, 64
    conv7 = BatchNormalization()(conv7)
    
    up8 = Conv2D(nfilter, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv1,up8], axis = 3)
    print("merge8 shape : ", merge8.shape)
    conv8 = Conv2D(nfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(nfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8) # 80, 80, 64, 32
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(nfilter, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    conv9 = Conv2D(nfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(nfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9) # 80, 80, 64, 32
    conv9 = BatchNormalization()(conv9)





    up10 = Conv2D(nfilter, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
    up10 = BatchNormalization()(up10)
    conv10 = Conv2D(int(nfilter/8), 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(2, 1, activation = None)(conv10)
    
    model = Model([inputs], [conv10])
    
    return model