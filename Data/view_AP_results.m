clear; close all; clc;

dataset_name = 'DIV2K_800_r1024_d128';

AP_dir_arr9   = sprintf('AP/%s/array9_iters10/Valid', dataset_name);
AP_dir_arr7   = sprintf('AP/%s/array7_iters10/Valid', dataset_name);
AP_dir_arr5   = sprintf('AP/%s/array5_iters10/Valid', dataset_name);
label_dir = sprintf('%s/Valid', dataset_name);

AP_dir9_file   = dir(fullfile(AP_dir_arr9,'*.mat'));
AP_dir7_file   = dir(fullfile(AP_dir_arr7,'*.mat'));
AP_dir5_file   = dir(fullfile(AP_dir_arr5,'*.mat'));
label_dir_file = dir(fullfile(label_dir,'*.mat'));

for i = 1
% for i = 1 : length(AP_dir1_file)
    load(sprintf('%s/%s', label_dir, label_dir_file(i).name));
    img_label = img;
    load(sprintf('%s/%s', AP_dir_arr9, AP_dir9_file(i).name));
    img_AP_arr9 = img_AP;
    load(sprintf('%s/%s', AP_dir_arr7, AP_dir7_file(i).name));
    img_AP_arr7 = img_AP;
    load(sprintf('%s/%s', AP_dir_arr5, AP_dir5_file(i).name));
    img_AP_arr5 = img_AP;
    
%     jimg(angle(img_label),   [-pi pi]);
%     jimg(angle(img_AP_arr9), [-pi pi]);
%     jimg(angle(img_AP_arr7), [-pi pi]);
    
%     figure; histogram(abs(img_label), 100);
%     figure; histogram(abs(img_AP_arr7), 100);
    
    figure; histogram(angle(img_label), 500);
    figure; histogram(angle(img_AP_arr9), 500);
    figure; histogram(angle(img_AP_arr7), 500);
    figure; histogram(angle(img_AP_arr5), 500);
    
end