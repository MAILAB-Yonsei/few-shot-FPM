clear; close all; clc;

epoch_interp = '0216';
epoch = '0326';

size_HR = 128;
size_LR = 64;
NA = 0.2;    
LEDgap = 0.5;
LEDheight = 10;
arraysize = 9;

model_name    = 'Baseline_Unet';
dataset_name  = 'DIV2K_800';
ind_test_st   = 351;

addpath('./utils');
load_dir = sprintf('HR%d_LR%d_NA%.2f_gap%.2f_height%.1f_arraysize_%d', size_HR, size_LR, NA, LEDgap, LEDheight, arraysize);

load(sprintf('../Developing/%s/output_HR_%s_interp/pred_img_%s.mat', model_name, load_dir, epoch_interp));
pred_img_interp = pred_img;
load(sprintf('../Developing/%s/output_HR_%s/pred_img_%s.mat', model_name, load_dir, epoch));

load(sprintf('../Data/Complex_3d_%s_resample%d/Complex_3d.mat', dataset_name, size_HR));
size_HR = size(img_3d, 1);

lr_from_hr_ft = zeros(size(img_3d));
lr_from_hr    = zeros(size(img_3d));
img_3d_ft     = zeros(size(img_3d));

for i = 1 : size(img_3d, 3)
    st  = (size_HR - size_LR)/2 + 1;
    fin = st + size_LR - 1;
    
    img_3d_ft(:,:,i) = fftshift(fft2(img_3d(:,:,i)));

    lr_from_hr_ft(st:fin,st:fin,i) = img_3d_ft(st:fin,st:fin,i);
    
    lr_from_hr(:,:,i) = ifft2(ifftshift(lr_from_hr_ft(:,:,i)));
end

jimg(abs(img_3d(:,:,ind_test_st:end)));
jimg(angle(img_3d(:,:,ind_test_st:end)), [0 1]);
jimg(abs(lr_from_hr(:,:,ind_test_st:end)));
jimg(angle(lr_from_hr(:,:,ind_test_st:end)),[0 1]);

recon_mag   = pred_img(:,:,:,1);
recon_mag   = permute(recon_mag,   [2 3 1]);
recon_phase = pred_img(:,:,:,2);
recon_phase = permute(recon_phase, [2 3 1]);

recon_mag_interp   = pred_img_interp(:,:,:,1);
recon_mag_interp   = permute(recon_mag_interp,   [2 3 1]);
recon_phase_interp = pred_img_interp(:,:,:,2);
recon_phase_interp = permute(recon_phase_interp, [2 3 1]);

jimg(recon_mag);
jimg(recon_phase, [0 1]);

jimg(recon_mag_interp);
jimg(recon_phase_interp, [0 1]);

lr_mag_psnr = psnr(abs(img_3d(:,:,ind_test_st)),abs(lr_from_hr(:,:,ind_test_st)))
recon_mag_psnr = psnr(abs(img_3d(:,:,ind_test_st)),double(recon_mag(:,:,1)))
recon_interp_mag_psnr = psnr(abs(img_3d(:,:,ind_test_st)),double(recon_mag_interp(:,:,1)))

lr_phase_psnr = psnr(angle(img_3d(:,:,ind_test_st)),angle(lr_from_hr(:,:,ind_test_st)))
recon_phase_psnr = psnr(angle(img_3d(:,:,ind_test_st)),double(recon_phase(:,:,1)))
recon_interp_phase_psnr = psnr(angle(img_3d(:,:,ind_test_st)),double(recon_phase_interp(:,:,1)))