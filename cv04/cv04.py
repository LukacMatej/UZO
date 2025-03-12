import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn 

def convertToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def loadImage(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image

def plotSpectrum(fft2, fft2_shift, result1, result2):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title("FFT2")
    plt.imshow(np.log(np.abs(fft2)),cmap='jet')
    plt.subplot(2, 2, 3)
    plt.imshow(np.log(np.abs(fft2_shift)),cmap='jet')
    plt.subplot(2, 2, 2)
    plt.title("Results")
    plt.imshow(result1,cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(result2,cmap='gray')
    plt.show()
    
def plotSpectrum2(fft2, fft2_shift):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("FFT2")
    plt.imshow(np.log(np.abs(fft2)),cmap='jet')
    plt.subplot(1, 2, 2)
    plt.imshow(np.log(np.abs(fft2_shift)),cmap='jet')
    plt.show()

def FiltDPHP(gray_image,fft2_shift):
    dp_filtr1 = loadImage("./cv04/cv04c_filtDP.bmp")
    dp_filtr2 = loadImage("./cv04/cv04c_filtDP1.bmp")
    conv_filt1 = convertFilt(dp_filtr1,gray_image,fft2_shift)
    conv_filt2 = convertFilt(dp_filtr2,gray_image,fft2_shift)
    imDP1 = np.fft.ifft2(np.fft.ifftshift(conv_filt1))
    imDP1 = np.abs(imDP1)
    imDP1 = imDP1 / np.max(imDP1)
    imDP2 = np.fft.ifft2(np.fft.ifftshift(conv_filt2))
    imDP2 = np.abs(imDP2)
    imDP2 = imDP2 / np.max(imDP2)
    
    hp_filtr1 = loadImage("./cv04/cv04c_filtHP.bmp")
    hp_filtr2 = loadImage("./cv04/cv04c_filtHP1.bmp")
    conv_filt3 = convertFilt(hp_filtr1,gray_image,fft2_shift)
    conv_filt4 = convertFilt(hp_filtr2,gray_image,fft2_shift)
    imDP3 = np.fft.ifft2(np.fft.ifftshift(conv_filt3))
    imDP3 = np.abs(imDP3)
    imDP3 = imDP3 / np.max(imDP3)
    imDP4 = np.fft.ifft2(np.fft.ifftshift(conv_filt4))
    imDP4 = np.abs(imDP4)
    imDP4 = imDP4 / np.max(imDP4)
    
    return imDP1,imDP2,imDP3,imDP4,conv_filt1,conv_filt2,conv_filt3,conv_filt4

def convertFilt(dp_filtr, gray_image,fft2_shift):
    dp_filtr_gray = convertToGray(dp_filtr)
    dp_filtr_resized = cv2.resize(dp_filtr_gray, (gray_image.shape[1], gray_image.shape[0]))
    fft2_shift_filtered = fft2_shift * dp_filtr_resized
    return fft2_shift_filtered

def convertToDCP(gray):
    return(dctn(gray))

def plotDct(gray,dcts):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Gary")
    plt.imshow(gray,cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("DctS")
    plt.imshow(np.log(np.abs(dcts)),cmap='jet')
    plt.show()

def plotDct6(gray1,dcts1,gray2,dcts2,gray3,dcts3):
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 2, 1)
    plt.title("Gary")
    plt.imshow(gray1,cmap='gray')
    plt.subplot(3, 2, 2)
    plt.title("DctS")
    plt.imshow(np.log(np.abs(dcts1)),cmap='jet')
    plt.subplot(3, 2, 3)
    plt.imshow(gray2,cmap='gray')
    plt.subplot(3, 2, 4)
    plt.imshow(np.log(np.abs(dcts2)),cmap='jet')
    plt.subplot(3, 2, 5)
    plt.imshow(gray3,cmap='gray')
    plt.subplot(3, 2, 6)
    plt.imshow(np.log(np.abs(dcts3)),cmap='jet')
    plt.show()

def limitDCT(dcts, limit):
    limited_dcts = np.zeros_like(dcts)
    limited_dcts[:limit, :limit] = dcts[:limit, :limit]
    return limited_dcts

def main():
    path_to_image = "./cv04/cv04c_robotC.bmp"
    image = loadImage(path_to_image)
    gray_image = convertToGray(image)
    fft2 = np.fft.fft2(gray_image)
    fft2_shift = np.fft.fftshift(fft2)
    plotSpectrum2(fft2,fft2_shift)
    imDP1,imDP2,imDP3,imDP4,conv_filt1,conv_filt2,conv_filt3,conv_filt4 = FiltDPHP(gray_image,fft2_shift)
    plotSpectrum(conv_filt1, conv_filt2, imDP1, imDP2)
    plotSpectrum(conv_filt3, conv_filt4, imDP3, imDP4)
    dctS = convertToDCP(gray_image)
    dctS10 = limitDCT(dctS,10)
    dctS30 = limitDCT(dctS,30)
    dctS50 = limitDCT(dctS,50)
    gray10 = idctn(dctS10)
    gray30 = idctn(dctS30)
    gray50 = idctn(dctS50)
    plotDct6(gray10,dctS10,gray30,dctS30,gray50,dctS50)
    
if __name__ == "__main__":
    main()