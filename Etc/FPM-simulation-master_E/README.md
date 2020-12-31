# FPM simulation

microscopy 이미지를 object로 사용하면 좋지만, 우선 모델의 테스트를 위해서는 아무 이미지나 인풋으로 사용해도 될것 같습니다.

Potential Input data link: https://data.vision.ee.ethz.ch/cvl/DIV2K/

오브젝트로 사용할 이미지를 아래의 objectAmplitude, phase 변수에 넣어주면 됩니다. 

코드 상의 아래 두 부분에 이미지를 넣어주면 복원 하고 싶은 object의 amplitude와 phase를 모델링 할 수 있습니다. 

<pre><code>
objectAmplitude = Image.open('./cameraman.tif')
phase = Image.open('./westconcordorthophoto.png')
</code></pre>

아래 코드에서 imSeqLowRes[:,:,tt] 부분이 촬영되는 이미지를 모델링 한 부분이고, 주석된 부분을 제거하면 저장됩니다.


<pre><code>
for tt in range (0,arraysize**2):
    kxc = int((n+1)/2+kx[0,tt]/dkx)
    kyc = int((m+1)/2+ky[0,tt]/dky)
    kxl=int((kxc-(n1-1)/2))
    kxh=int((kxc+(n1-1)/2))
    kyl=int((kyc-(m1-1)/2))
    kyh=int((kyc+(m1-1)/2))
    imSeqLowFT = ((m1/m)**2) * objectFT[kyl:kyh+1,kxl:kxh+1]* CTF
    imSeqLowRes[:,:,tt] = np.absolute(np.fft.ifft2(np.fft.ifftshift(imSeqLowFT)))
    ##촬영되는 이미지를 저장하려면
    #plt.imsave('image_'+str(tt)+'.png',imSeqLowRes[:,:,1])
</code></pre>

