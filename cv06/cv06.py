import numpy as np
import cv2
import matplotlib.pyplot as plt

def loadImage(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image

def convertToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def plotSpectrum(original, spectrum1, different, spectrum2):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title("Spectrum")
    plt.imshow(np.log(np.abs(spectrum1)), cmap='jet')
    plt.colorbar(label='Intensity', pad=0.1)
    plt.subplot(2, 2, 3)
    plt.title("Filtered")
    plt.imshow(different, cmap='jet')
    plt.colorbar(label='Intensity', pad=0.1)
    plt.subplot(2, 2, 4)
    plt.title("Spectrum")
    plt.imshow(np.log(np.abs(spectrum2)), cmap='jet')
    plt.colorbar(label='Intensity', pad=0.1)
    plt.show()
    
def plotKirsch(robot_s_gray):
    fft1 = np.fft.fft2(robot_s_gray)
    fft1_shift = np.fft.fftshift(fft1)
    laplac = applyKirsch(robot_s_gray)
    fft2 = np.fft.fft2(laplac)
    fft2_shift = np.fft.fftshift(fft2)
    plotSpectrum(robot_s_gray,fft1_shift,laplac,fft2_shift)
    
def plotSobel(robot_s_gray):
    fft1 = np.fft.fft2(robot_s_gray)
    fft1_shift = np.fft.fftshift(fft1)
    laplac = applySobel(robot_s_gray)
    fft2 = np.fft.fft2(laplac)
    fft2_shift = np.fft.fftshift(fft2)
    plotSpectrum(robot_s_gray,fft1_shift,laplac,fft2_shift)

def applyKirsch(image):
    kernel_x = np.array([[-5, 3, 3], 
                         [-5, 0, 3], 
                         [-5, 3, 3]])
    
    kernel_y = np.array([[3,3, 3], 
                         [3, 0, 3], 
                         [-5, -5, -5]])
    
    gradient_x = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_y)
    
    image_sob = np.sqrt(gradient_x**2 + gradient_y**2)
        
    return image_sob

def applySobel(image):
    kernel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]])
    
    kernel_y = np.array([[1, 2, 1], 
                         [0, 0, 0], 
                         [-1, -2, -1]])
    
    gradient_x = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, kernel=kernel_y)
    
    image_sob = np.sqrt(gradient_x**2 + gradient_y**2)
        
    return image_sob


def plotLaplacian(image_gray):
    fft1 = np.fft.fft2(image_gray)
    fft1_shift = np.fft.fftshift(fft1)
    laplacian = applyLaplacian(image_gray)
    fft2 = np.fft.fft2(laplacian)
    fft2_shift = np.fft.fftshift(fft2)
    plotSpectrum(image_gray, fft1_shift, laplacian, fft2_shift)

def applyLaplacian(image):
    kernel2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    laplacian = cv2.filter2D(image, cv2.CV_16S, kernel2)
    return laplacian

def main():
    robot_c_path = "./cv06/cv06_robotC.bmp"
    robot_c = loadImage(robot_c_path)
    robot_c_gray = convertToGray(robot_c)
    plotLaplacian(robot_c_gray)
    plotSobel(robot_c_gray)
    plotKirsch(robot_c_gray)

    
if __name__ == "__main__":
    main()
