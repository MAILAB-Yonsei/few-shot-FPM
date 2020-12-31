import numpy as np

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

def extract_indices_CTF(opt):
    waveLength = 0.518e-6
    k0 = 2*np.pi/waveLength
    psize = opt.spsize / (opt.size_HR / opt.size_LR);

    xlocation = np.zeros((1,opt.array_size**2));
    ylocation = np.zeros((1,opt.array_size**2));
            
    for it in range (0, opt.array_size):# from top left to bottom right
        for k in range(0, opt.array_size):
            xlocation[0, k + opt.array_size*(it)] = (((1 - opt.array_size)/2) + k) * opt.LEDgap
            ylocation[0, k + opt.array_size*(it)] = ((opt.array_size - 1)/2 - (it)) * opt.LEDgap
    
    kx_relative = -np.sin(np.arctan(xlocation/opt.LEDheight));
    ky_relative = -np.sin(np.arctan(ylocation/opt.LEDheight));
    
    m = opt.size_HR
    n = opt.size_HR
    
    m1 = int(m/(opt.spsize/psize)) # image size of the final output
    n1 = int(n/(opt.spsize/psize))  

    kx = k0 * kx_relative;                 
    ky = k0 * ky_relative;
    dkx = 2*np.pi/(psize*n);
    dky = 2*np.pi/(psize*m);
    cutoffFrequency = opt.NA * k0;
    kmax = np.pi/opt.spsize
    kxm=np.zeros((m1,n1))
    kym=np.zeros((m1,n1))
    
    instx=np.linspace(-kmax, kmax, m1)
    insty=np.linspace(-kmax, kmax, n1)
    [kxm, kym]= np.meshgrid(instx, insty)
    
    CTF = ((kxm**2+kym**2) < cutoffFrequency**2) #pupil function circ(kmax)
    seq = gseq(opt.array_size)
    
    indices_e = {}; indices_f = {}
    indices_e["kxl"] = [];  indices_e["kxh"] = []; indices_e["kyl"] = []; indices_e["kyh"] = []
    indices_f["kxl"] = [];  indices_f["kxh"] = []; indices_f["kyl"] = []; indices_f["kyh"] = []
    
    for i in range (0, opt.array_size**2):
        kxc = int((n+1)/2+kx[0,i]/dkx)
        kyc = int((m+1)/2+ky[0,i]/dky)
        indices_e["kxl"].append(int((kxc-(n1-1)/2)))
        indices_e["kxh"].append(int((kxc+(n1-1)/2)))
        indices_e["kyl"].append(int((kyc-(m1-1)/2)))
        indices_e["kyh"].append(int((kyc+(m1-1)/2)))
        
        i2  = int(seq[i]-1)
        kxc = int((n+1)/2+kx[0,i2]/dkx)
        kyc = int((m+1)/2+ky[0,i2]/dky)
        indices_f["kxl"].append(int((kxc-(n1-1)/2)))
        indices_f["kxh"].append(int((kxc+(n1-1)/2)))
        indices_f["kyl"].append(int((kyc-(m1-1)/2)))
        indices_f["kyh"].append(int((kyc+(m1-1)/2)))
        
    return CTF, indices_e, indices_f, seq
    
def extract_FPM_images(img_obj, opt, indices_e, CTF):
    img_FPM      = np.zeros((opt.array_size ** 2, opt.size_LR, opt.size_LR))
    img_FPM_sqrt = np.zeros((opt.array_size ** 2, opt.size_LR, opt.size_LR))
    img_FPM_comp = np.zeros((opt.array_size ** 2, opt.size_LR, opt.size_LR), dtype='complex_')
    img_FPM_ri   = np.zeros((opt.array_size ** 2 * 2, opt.size_LR, opt.size_LR))
    
    FT_obj = np.fft.fftshift(np.fft.fft2(img_obj));
    
    for i in range (0, opt.array_size**2):
        FT_low_res = FT_obj[indices_e["kyl"][i]:indices_e["kyh"][i] + 1, indices_e["kxl"][i]:indices_e["kxh"][i] + 1] * CTF
        one_slice = np.fft.ifft2(np.fft.ifftshift(FT_low_res))
        
        img_FPM_comp[i, :, :] = one_slice
        img_FPM_sqrt[i, :, :] = np.absolute(one_slice)
        
        img_FPM[i, :, :] = np.absolute(one_slice) ** 2
        img_FPM_ri[i*2,   :, :] = np.real(one_slice)
        img_FPM_ri[i*2+1, :, :] = np.imag(one_slice)
      
    return img_FPM, img_FPM_ri, img_FPM_sqrt, img_FPM_comp

if __name__ == "__main__":
    for i in range(25):
        print(int(gseq(5)[i])-1)