clear; close all; clc;

load('example.mat')
fpm = permute(fpm, [2, 3, 1]);

fpm_mag = abs(fpm);
fpm_phase = angle(fpm);

ft_fpm = zeros(size(fpm));
ft_fpm_mag = zeros(size(fpm));
ft_fpm_phase = zeros(size(fpm));

for i = 1 : 25
    ft_fpm_mag(:,:,i) = fftshift(fft2(fpm_mag(:,:,i)));
    ft_fpm_phase(:,:,i) = fftshift(fft2(fpm_phase(:,:,i)));
    ft_fpm(:,:,i) = fftshift(fft2(fpm(:,:,i)));
end

jimg(abs(ft_fpm).^0.2, [0 4])
jimg(abs(ft_fpm_mag).^0.2, [0 4])
jimg(abs(ft_fpm_phase).^0.2, [0 4])

jimg(fpm, [0 1])
jimg(fpm_mag, [0 1])

fpm_mag_mean = mean(fpm_mag(:,:,14:25), 3);
jimg(fpm_mag_mean);
jimg(img)