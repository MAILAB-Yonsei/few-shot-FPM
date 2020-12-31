clear; close all; clc;

rsize = 1024;
dsize = 128;

num_data = 800;

mkdir(sprintf('DIV2K_800_r%d_d%d', rsize, dsize));
    
for i = 1 : 2 : num_data
    fprintf(sprintf('%d/%d\n',i,num_data));
    
    img_hr_mag   = rgb2gray(imread(sprintf('../../Data/DIV2K_train_HR/%04d.png', i)));
    img_hr_mag   = rescale(imresize(img_hr_mag, [rsize rsize]));
    img_hr_phase = rgb2gray(imread(sprintf('../../Data/DIV2K_train_HR/%04d.png', i + 1)));
    img_hr_phase = rescale(imresize(img_hr_phase, [rsize rsize]));
    img_hr = img_hr_mag .* exp(1i * pi * (img_hr_phase * 2 - 1));
    
    patch_ind = 1;
    for k = 1 : rsize / dsize
        for kk = 1 : rsize / dsize
            img = img_hr(dsize*(k-1)+1 : dsize*(k), dsize*(kk-1)+1 : dsize*(kk));
            save(sprintf('DIV2K_800_r%d_d%d/%04d_%03d.mat', rsize, dsize, (i + 1)/2, patch_ind), 'img');
            patch_ind = patch_ind + 1;
        end
    end
end