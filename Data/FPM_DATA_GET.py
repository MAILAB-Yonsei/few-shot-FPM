import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import cv2

check_data = True # Generate only first data, and view the data

# parameter for Imaging System
size_HR    = 128
size_LR    = 32
NA = 0.2      # NA of 대물렌즈
LEDgap     = 12  # gap between adjacent LEDs
LEDheight  = 50 # distance bewteen the LED matrix and the sample
arraysize  = 5

spsize = 1.5e-6;        #sampling pixel size of the camera

# size_LR 32일 때는 LEDgap = 2.0
# size_LR 64일 때는 LEDgap = 0.5

num_train  = 350
num_test   = 40

if check_data == True:
    num_data = 1
else:
    num_data = num_train + num_test

save_dir = 'HR%d_LR%d_NA%.2f_gap%.2f_height%.1f_arraysize_%d' % (size_HR, size_LR, NA, LEDgap, LEDheight, arraysize)
os.makedirs(save_dir, exist_ok=True)

LR_Amp_Train  = np.zeros((num_train, size_LR, size_LR, arraysize ** 2))
LR_Comp_Train = np.zeros((num_train, size_LR, size_LR, arraysize ** 2 * 2))
LR_Comp2_Train= np.zeros((num_train, size_LR, size_LR, arraysize ** 2 * 2))

LR_Amp_Test   = np.zeros((num_test,  size_LR, size_LR, arraysize ** 2))
LR_Comp_Test  = np.zeros((num_test,  size_LR, size_LR, arraysize ** 2 * 2))
LR_Comp2_Test = np.zeros((num_test,  size_LR, size_LR, arraysize ** 2 * 2))

HR_Real_Train = np.zeros((num_train, size_HR, size_HR))
HR_Imag_Train = np.zeros((num_train, size_HR, size_HR))

HR_Amp_Test   = np.zeros((num_test,  size_HR, size_HR))
HR_Phase_Test = np.zeros((num_test,  size_HR, size_HR))

HR_Amp_Train  = np.zeros((num_train, size_HR, size_HR))
HR_Phase_Train= np.zeros((num_train, size_HR, size_HR))

HR_Real_Test  = np.zeros((num_test,  size_HR, size_HR))
HR_Imag_Test  = np.zeros((num_test,  size_HR, size_HR))

LR_Amp_Train_rescale  = np.zeros((num_train, size_HR, size_HR, arraysize ** 2))
LR_Amp_Test_rescale   = np.zeros((num_test,  size_HR, size_HR, arraysize ** 2))

for ii in range(num_data):
    objectAmplitude = scipy.io.loadmat('./Amplitude_DIV2K_800_resample%d/%04d_Amplitude.mat' % (size_HR, ii + 1))['img']
    
    phase = scipy.io.loadmat('./Phase_DIV2K_800_resample%d/%04d_Phase.mat' % (size_HR, ii + 1))['img']
    phase = np.array(phase)
    
    object_function = np.multiply(objectAmplitude,np.exp(2j*np.pi*phase))
    
    #%%
    #parameter for illumination
    waveLength = 0.518e-6;    #사용된 빛의 종류
    k0 = 2*np.pi/waveLength;  # wavevector
    
    #parameter for simulation (Ground Truth)
#    psize = spsize / 2;      #the pixel size of the final reconstructed super-resolution image
    psize = spsize / (size_HR/size_LR);
    
    #%%   
    xlocation = np.zeros((1,arraysize**2));
    ylocation = np.zeros((1,arraysize**2));
    
    for i in range (0,arraysize):# from top left to bottom right
        for k in range(0,arraysize):
            xlocation[0, k + arraysize*(i)] = (((1 - arraysize)/2) + k) * LEDgap
            ylocation[0, k + arraysize*(i)] = ((arraysize - 1)/2 - (i)) * LEDgap
    
    ## Relative Fourier Domain location of LED array
    kx_relative = -np.sin(np.arctan(xlocation/LEDheight));  
    ky_relative = -np.sin(np.arctan(ylocation/LEDheight));   
    
    #%%
       
    [m,n] = object_function.shape; #image size of high resolution object
    m1 = int(m/(spsize/psize))     #image size of the final output
    n1 = int(n/(spsize/psize))     
    
    imSeqLowRes = np.zeros((m1, n1, arraysize**2)); #the final low resolution image sequence
    imSeqLowResComp2ch = np.zeros((m1, n1, arraysize**2 * 2)); #the final low resolution image sequence
    imSeqLowResComp2ch2= np.zeros((m1, n1, arraysize**2 * 2)); #the final low resolution image sequence
    kx = k0 * kx_relative;                 
    ky = k0 * ky_relative;
    dkx = 2*np.pi/(psize*n);
    dky = 2*np.pi/(psize*m);
    cutoffFrequency = NA * k0;
    kmax = np.pi/spsize
    kxm=np.zeros((m1,n1))
    kym=np.zeros((m1,n1))
    
    instx=np.linspace(-kmax,kmax,m1)
    insty=np.linspace(-kmax,kmax,n1)
    [kxm, kym]= np.meshgrid(instx,insty)
    CTF = ((kxm**2+kym**2) < cutoffFrequency**2); #pupil function circ(kmax);
    
    # plt.imshow(CTF,cmap='gray')
    # plt.show()
    
    objectFT = np.fft.fftshift(np.fft.fft2(object_function));
    
    ## Low Resolution 묶음 만들기
    for tt in range (0,arraysize**2):
        kxc = int((n+1)/2+kx[0,tt]/dkx)
        kyc = int((m+1)/2+ky[0,tt]/dky)
        kxl=int((kxc-(n1-1)/2))
        kxh=int((kxc+(n1-1)/2))
        kyl=int((kyc-(m1-1)/2))
        kyh=int((kyc+(m1-1)/2))
        
        imSeqLowFT = objectFT[kyl:kyh+1,kxl:kxh+1]* CTF
        
        imSeqLowResComp = np.fft.ifft2(np.fft.ifftshift(imSeqLowFT))
        
        imSeqLowRes[:,:,tt] = np.absolute(imSeqLowResComp)**2 # why square??
        imSeqLowResComp2ch[:,:,tt*2]   = np.real(imSeqLowResComp)
        imSeqLowResComp2ch[:,:,tt*2+1] = np.imag(imSeqLowResComp)
        imSeqLowResComp2ch2[:,:,tt*2]  = np.absolute(imSeqLowResComp)
        imSeqLowResComp2ch2[:,:,tt*2+1]= np.angle(imSeqLowResComp)
    #     ## 촬영되는 이미지를 저장하려면
    #     plt.imsave('./tmp/image_'+str(tt)+'.png',imSeqLowRes[:,:,tt])
       
    print('%d / %d' % (ii, num_data))
    
    if ii < num_train:
        LR_Amp_Train[ii,:,:,:]  = imSeqLowRes
        LR_Comp_Train[ii,:,:,:] = imSeqLowResComp2ch
        LR_Comp2_Train[ii,:,:,:]= imSeqLowResComp2ch2
        
        HR_Amp_Train[ii,:,:]    = objectAmplitude
        HR_Phase_Train[ii,:,:]  = phase
        
        HR_Real_Train[ii,:,:]   = object_function.real
        HR_Imag_Train[ii,:,:]   = object_function.imag
        
    else:
        LR_Amp_Test[ii - num_train, :,:,:]  = imSeqLowRes
        LR_Comp_Test[ii - num_train,:,:,:]  = imSeqLowResComp2ch
        LR_Comp2_Test[ii - num_train,:,:,:] = imSeqLowResComp2ch2
        
        HR_Amp_Test[ii   - num_train,:,:]   = objectAmplitude
        HR_Phase_Test[ii - num_train,:,:]   = phase
        
        HR_Real_Test[ii  - num_train,:,:]   = object_function.real
        HR_Imag_Test[ii  - num_train,:,:]   = object_function.imag
       
for ii in range(num_train):
    for jj in range(arraysize ** 2):
        LR_Amp_Train_rescale[ii, :, :, jj] = cv2.resize(LR_Amp_Train[ii, :, :, jj], dsize=(size_HR, size_HR), interpolation=cv2.INTER_CUBIC)
    
for ii in range(num_test):
    for jj in range(arraysize ** 2):
        LR_Amp_Test_rescale[ii, :, :, jj]  = cv2.resize(LR_Amp_Test[ii, :, :, jj],  dsize=(size_HR, size_HR), interpolation=cv2.INTER_CUBIC)


np.save('./%s/LR_Amp_Train.npy' % save_dir ,  LR_Amp_Train)
np.save('./%s/LR_Comp_Train.npy' % save_dir , LR_Comp_Train)
np.save('./%s/LR_Comp2_Train.npy'% save_dir , LR_Comp2_Train)

np.save('./%s/HR_Amp_Train.npy' % save_dir ,  HR_Amp_Train)
np.save('./%s/HR_Phase_Train.npy' % save_dir, HR_Phase_Train)
np.save('./%s/HR_Real_Train.npy' % save_dir , HR_Real_Train)
np.save('./%s/HR_Imag_Train.npy' % save_dir , HR_Imag_Train)

np.save('./%s/LR_Amp_Test.npy' % save_dir ,   LR_Amp_Test)
np.save('./%s/LR_Comp_Test.npy' % save_dir ,  LR_Comp_Test)
np.save('./%s/LR_Comp2_Test.npy'% save_dir ,  LR_Comp2_Test)

np.save('./%s/HR_Amp_Test.npy' % save_dir ,   HR_Amp_Test)
np.save('./%s/HR_Phase_Test.npy' % save_dir , HR_Phase_Test)
np.save('./%s/HR_Real_Test.npy' % save_dir ,  HR_Real_Test)
np.save('./%s/HR_Imag_Test.npy' % save_dir ,  HR_Imag_Test)

np.save('./%s/LR_Amp_Train_interp.npy' % save_dir,  LR_Amp_Train_rescale)
np.save('./%s/LR_Amp_Test_interp.npy'  % save_dir,  LR_Amp_Test_rescale)

# Low Resolution Images
plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(imSeqLowRes[:,:,0],cmap='gray')
plt.subplot(2,2,2)
plt.imshow(imSeqLowRes[:,:,(arraysize ** 2) // 2 - 1], cmap='gray')
plt.subplot(2,2,3)
plt.imshow(imSeqLowRes[:,:,(arraysize ** 2) // 2    ], cmap='gray')
plt.subplot(2,2,4)
plt.imshow(imSeqLowRes[:,:,-1],cmap='gray')
plt.show()

if check_data == True:
    plt.figure(2)
    plt.imshow(np.squeeze(HR_Amp_Train[ii,:,:]))
    plt.figure(3)
    plt.imshow(np.squeeze(HR_Phase_Train[ii,:,:]))

#%%
def gseq(arraysize):

    n=(arraysize+1)/2;
    sequence=np.zeros((2,arraysize**2))
    sequence[0,0]=n;
    sequence[1,0]=n;			
    dx=+1;
    dy=-1;
    stepx=+1;
    stepy=-1;
    direction=+1;
    counter=0;
  
    for i in range (1,arraysize**2):
        counter=counter+1;
        if (direction==+1):
            sequence[0,i]=sequence[0,i-1]+dx;
            sequence[1,i]=sequence[1,i-1];
            if (counter==abs(stepx)):
                counter=0;
                direction=direction*-1;
                dx=dx*-1;
                stepx=stepx*-1;
                if stepx>0:
                    stepx=stepx+1;
                else:
                    stepx=stepx-1;
    
        else:
            sequence[0,i]=sequence[0,i-1];
            sequence[1,i]=sequence[1,i-1]+dy;
            if (counter==abs(stepy)):
                counter=0;
                direction=direction*-1;
                dy=dy*-1;
                stepy=stepy*-1;
                if (stepy>0):
                    stepy=stepy+1;
                else:
                    stepy=stepy-1;
    
    seq=(sequence[0,:]-1)*arraysize+sequence[1,:]
    return seq

#%% Alternate Projection

seq=gseq(arraysize)

# initialization
objectRecover = np.ones((m,n));
objectRecoverFT = np.fft.fftshift(np.fft.fft2(objectRecover));


# plt.imshow(CTF,cmap='gray')
# plt.show

loop = 20;
for tt in range (0,loop):
    for i3 in range(0,arraysize**2):
        i2=int(seq[i3]-1)
        
        kxc = int((n+1)/2 + kx[0, i2]/dkx);
        kyc = int((m+1)/2 + ky[0, i2]/dky);
        kxl = int((kxc - (n1-1)/2))
        kxh = int((kxc + (n1-1)/2))
        kyl = int((kyc - (m1-1)/2))
        kyh = int((kyc + (m1-1)/2))
        
#         lowResFT = ((m1/m)**2) * objectRecoverFT[kyl:kyh+1,kxl:kxh+1]* CTF
        lowResFT = objectRecoverFT[kyl:kyh+1, kxl:kxh+1] * CTF
        
        im_lowRes = np.fft.ifft2(np.fft.ifftshift(lowResFT));
#         im_lowRes = (m/m1)**2 * np.multiply(imSeqLowRes[:,:,i2],np.exp(1j*np.angle(im_lowRes)))
        im_lowRes = np.sqrt(imSeqLowRes[:,:,i2]) * (im_lowRes) / abs(im_lowRes) # Phase만 update하여 곱하는 과정 (im_lowRes)/abs(im_lowRes) 가 phase image
        
        lowResFT=np.fft.fftshift(np.fft.fft2(im_lowRes)) * CTF;
        
        objectRecoverFT[kyl:kyh+1, kxl:kxh+1] = np.multiply((1-CTF), objectRecoverFT[kyl:kyh+1, kxl:kxh+1]) + lowResFT;            
        if (tt == 0) and (i3 == 0 or i3 == 1 or i3 == 2 or i3 == 3 or i3 == 4):
            plt.figure(4)
            plt.imshow(np.log(0.00000001+np.absolute(objectRecoverFT)),cmap='gray')
            # plt.imshow(abs(np.fft.ifft2(np.fft.ifftshift(objectRecoverFT[kyl:kyh+1,kxl:kxh+1]*CTF))))

objectRecover=np.fft.ifft2(np.fft.ifftshift(objectRecoverFT));

plt.figure(figsize=(20, 20))
plt.subplot(221)
plt.imshow(np.absolute(objectRecover),cmap='gray')
plt.title('recovered complex field (mag)')
plt.colorbar()
plt.subplot(222)
plt.imshow(np.angle(objectRecover),cmap='gray');
# plt.clim([0, np.pi])
plt.title('recovered complex field (phase)')
plt.colorbar()
plt.subplot(223)
plt.imshow(imSeqLowRes[:,:,int((arraysize**2-1)/2)],cmap='gray');
plt.title('captured low_res image')
plt.colorbar()
plt.subplot(224)
plt.imshow((np.log(0.01+np.absolute(objectRecoverFT))),cmap='gray');
plt.title('FT of complex field')
plt.show()

plt.imshow(CTF)