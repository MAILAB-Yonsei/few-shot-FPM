
i=2
for i = 1 : 800
    
    i
    img = imread(sprintf('./DIV2K_train_HR_sample/%04d.png',i));
    img = rgb2gray(img);
    img = imresize(img , [256 256]);  
    img= rescale(img);

    ind = mod(i,2);
    rid = round(i/2);    
    
    if ind == 0 

        save( sprintf('./Phase/%04d_Phase.mat', rid) , 'img');

    end
   
    if ind ~= 0 
        save( sprintf('./Intensity/%04d_Intensity.mat', rid) , 'img');
        

    end
end





i=701
for i = 1 : 800
    
    i
    img2 = imread(sprintf('./DIV2K_train_HR_sample/%04d.png',i));
    img2 = rgb2gray(img2);
    img2 = imresize(img2 , [128 128]);  
    img2= rescale(img2);

    ind = mod(i,2);
    rid = round(i/2);    
    
    if ind == 0 

        save( sprintf('./Phase/%04d_Phase.mat', rid) , 'img');

    end
   
    if ind ~= 0 
        save( sprintf('./Intensity/%04d_Intensity.mat', rid) , 'img');
        

    end
end


img = imresize(img2, [256 256]);

jimg(img);
size(pred_img);


A = pred_img(:,:,:,1);
A=permute(A, [ 2 3 1 ]);
B = pred_img(:,:,:,2);

B=permute(B, [ 2 3 1 ]);
jimg(A);
jimg(B);

TT = A

jimg(img);

jimg(B);

B= B+max(B);

im_comp=zeros(256,256,40);


for ii = 1:40
    im_comp(:,:,ii) = A(:,:,ii) + 1i*B(:,:,ii);
end

obj=zeros(256,256,40);
for ii = 1:40
    obj(:,:,ii) = A(:,:,ii) .* exp(1i*B(:,:,ii));
end



jimg(abs(obj));

jimg(angle(obj));


img3 = img2.*exp(1j.*img);