clear; close all; clc;

load('output_seg_real_imag/pred_img.mat');
load('./Complex_3d/Complex_3d.mat');

jimg(img_3d(:,:,351:end));

recon_real = pred_img(:,:,:,1);
recon_real = permute(recon_real, [2 3 1]);
recon_imag = pred_img(:,:,:,2);
recon_imag = permute(recon_imag, [2 3 1]);

jimg(recon_real);
jimg(recon_imag);

recon_comp = recon_real + 1i*recon_imag;

jimg(recon_comp);