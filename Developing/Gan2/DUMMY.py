
import numpy as np



def Phase_shift(AP_output, GT_phase):
    
    
    AP_shifted = np.zeros(AP_output.shape)
    
    AP_output_comp = (AP_output[:,:,:,0] + 1j*AP_output[:,:,:,1]).cpu()
    
    
    
    GT_comp = (GT_phase[:,:,:,0] + 1j*GT_phase[:,:,:,1]).cpu()


    for k in range(AP_output.shape[0]):
        AP_hist_dum = np.histogram( np.angle(AP_output_comp[k,:,:]).ravel(),   bins=256 , range = ( -np.pi , np.pi   ) ) 
        GT_hist_dum = np.histogram( np.angle(GT_comp[k,:,:]).ravel(),  bins=256 , range= ( -np.pi , np.pi   )) 

        AP_output_AMP_dum = np.absolute(AP_output_comp[k,:,:])
        AP_output_PHASE_dum = np.angle(AP_output_comp[k,:,:])
    
    
        AP_hist_dum_stack = np.concatenate([AP_hist_dum[0]  ,  AP_hist_dum[0] ] )
        
        
        max_correlations = 0
        
        
        z=59
        
        
        
        
        for z in range(256):
            correlations = np.sum(  (GT_hist_dum[0] * AP_hist_dum_stack[0+z:256+z]))
            
            if correlations > max_correlations :
                max_correlations = correlations
                
                
                AP_output_PHASE_shifted_dum = AP_output_PHASE_dum
                    
                
                AP_output_PHASE_shifted_dum[ AP_output_PHASE_shifted_dum < AP_hist_dum[1][z]  ] += 6.28
                
                
                AP_output_PHASE_shifted_dum -=(  AP_hist_dum[1][z] +3.14)
                
                
                
                # plt.imshow(AP_output_PHASE_shifted_dum)
                
                # np.max(AP_output_PHASE_shifted_dum)
                # np.min(AP_output_PHASE_shifted_dum)
                
                # np.max(np.angle(GT_comp[k,:,:]))
                # np.min(np.angle(GT_comp[k,:,:]))
                
                
                
                
               
                
        AP_shift_comp = np.multiply(AP_output_AMP_dum , np.exp(1j * AP_output_PHASE_shifted_dum)  )
        
        
        AP_shifted[k,:,:,0] = AP_shift_comp.real
        AP_shifted[k,:,:,1] = AP_shift_comp.imag
        
        AP_shifted = torch.tensor(AP_shifted)
    return AP_shifted
        
        


def Phase_shift_UV(AP_output, GT_phase) :
    
    
    
    AP_output_U = torch.div(AP_output , torch.clamp(   torch.sqrt(AP_output[:,:,:,0]**2 + AP_output[:,:,:,1]**2 )  , min=1e-6 )[:,:,:,None]).cpu()
    AP_output_mean = torch.sum(AP_output_U , dim=[-2, -3])/(AP_output_U.shape[-2]*AP_output_U.shape[-3])
    
    
    AP_output_mean_unit= torch.div( AP_output_mean ,  torch.clamp(  torch.sqrt(AP_output_mean[:,0]**2 + AP_output_mean[:,1]**2) , min = 1e-6)[:,None]  )
    
    
    
    AP_rotated = torch.zeros(AP_output.shape)
    
    for z in range( AP_output_mean.shape[0]):
        # if AP_output_mean[z, 1] >0 :
        AP_rotated[z,:,:,0] =  AP_output[z,:,:,0]*AP_output_mean_unit[z,0] + AP_output[z,:,:,1]*AP_output_mean_unit[z,1]
        AP_rotated[z,:,:,1] =  AP_output[z,:,:,0]*AP_output_mean_unit[z,1] - AP_output[z,:,:,1]*AP_output_mean_unit[z,0]
            
        # else:
        #     AP_rotated[z,:,:,0] =  AP_output[z,:,:,0]*AP_output_mean_unit[z,0] - AP_output[z,:,:,1]*AP_output_mean_unit[z,1]  
        #     AP_rotated[z,:,:,1] =  AP_output[z,:,:,0]*AP_output_mean_unit[z,1] + AP_output[z,:,:,1]*AP_output_mean_unit[z,0]  
    
    
    
    
    
    GT_U         = torch.div(GT_phase ,  torch.clamp(   torch.sqrt(GT_phase[:,:,:,0]**2 + GT_phase[:,:,:,1]**2   )  , min=1e-6  )[:,:,:,None])
    GT_mean        = torch.sum(GT_U , dim=[-2, -3]) /(GT_U.shape[-2]*GT_U.shape[-3])
    

    GT_output_mean_unit= torch.div( GT_mean ,  torch.clamp(  torch.sqrt(GT_mean[:,0]**2 + GT_mean[:,1]**2) , min = 1e-6)[:,None]  )
    
    GT_rotated = torch.zeros(GT_phase.shape)
    
    for z in range( GT_mean.shape[0]):
        # if GT_mean[z, 1] >0 :
        GT_rotated[z,:,:,0] =  GT_phase[z,:,:,0]*GT_output_mean_unit[z,0] + GT_phase[z,:,:,1]*GT_output_mean_unit[z,1]
        GT_rotated[z,:,:,1] =  GT_phase[z,:,:,0]*GT_output_mean_unit[z,1] - GT_phase[z,:,:,1]*GT_output_mean_unit[z,0]  
        
        # else:
        #     GT_rotated[z,:,:,0] =  GT_phase[z,:,:,0]*GT_output_mean_unit[z,0] - GT_phase[z,:,:,1]*GT_output_mean_unit[z,1] 
        #     GT_rotated[z,:,:,1] =  GT_phase[z,:,:,0]*GT_output_mean_unit[z,1] + GT_phase[z,:,:,1]*GT_output_mean_unit[z,0] 
    
    # GT_rotated[z,:,:,1].mean()
    
    
    return AP_rotated, GT_rotated, AP_output_mean_unit , GT_output_mean_unit



