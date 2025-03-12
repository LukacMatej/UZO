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

def getHistogram(image):
    return cv2.calcHist([image], [0], None, [256], [0, 256])

def plotSpectrum(original, spectrum1, average, spectrum2, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title("Spectrum")
    plt.imshow(np.log(np.abs(spectrum1)), cmap='jet')
    plt.subplot(2, 2, 3)
    plt.title(title)
    plt.imshow(average, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title("Spectrum")
    plt.imshow(np.log(np.abs(spectrum2)), cmap='jet')
    plt.show()

def simpleAveraging(image):
    kernel = np.ones((3, 3), np.float32) / 9
    return cv2.filter2D(image, -1, kernel)

def rotatingMaskFilter(image):
    height, width = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    masks = [
        np.array([ 
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]),
        np.array([
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 1]
        ]),
        np.array([
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1]
        ]),
        np.array([
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1]
        ]),
        np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]),
        np.array([
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ]),
        np.array([
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0]
        ]),
        np.array([
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0]
        ]),
        np.array([
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1]
        ])
    ]
    
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
    
    for i in range(height):
        for j in range(width):
            min_variance = float('inf')
            best_mean = 0
            
            for mask in masks:
                neighborhood = padded_image[i:i+3, j:j+3]
                mask_values = neighborhood[mask == 1]
                variance = np.var(mask_values)
                if variance < min_variance:
                    min_variance = variance
                    best_mean = np.mean(mask_values)
            result[i, j] = best_mean
    
    return result.astype(np.uint8)

def medianFilter(image, kernel_shape):
    padded_image = np.pad(image, ((kernel_shape[0]//2, kernel_shape[0]//2), (kernel_shape[1]//2, kernel_shape[1]//2)), mode='reflect')
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_shape[0], j:j+kernel_shape[1]]
            result[i, j] = np.median(region)
    return result

def plot(robot_s, robot_s_gray):
    fft1 = np.fft.fft2(robot_s_gray)
    fft1_shift = np.fft.fftshift(fft1)
    robot_s_average = simpleAveraging(robot_s_gray)    
    fft2 = np.fft.fft2(robot_s_average)
    fft2_shift = np.fft.fftshift(fft2)
    plotSpectrum(robot_s,fft1_shift,robot_s_average, fft2_shift, "Average")

    robot_s_average_rotation = rotatingMaskFilter(robot_s_gray)
    fft3 = np.fft.fft2(robot_s_average_rotation)
    fft3_shift = np.fft.fftshift(fft3)
    plotSpectrum(robot_s,fft1_shift,robot_s_average_rotation, fft3_shift, "Average, rotation mask")
    
    robot_s_median = medianFilter(robot_s_gray, (3, 3))
    fft4 = np.fft.fft2(robot_s_median)
    fft4_shift = np.fft.fftshift(fft4)
    plotSpectrum(robot_s,fft1_shift,robot_s_median, fft4_shift, "Median")

def main():
    robot_s_path = "./cv05/cv05_robotS.bmp"
    pss_path = "./cv05/cv05_PSS.bmp"
    robot_s = loadImage(robot_s_path)
    pss = loadImage(pss_path)
    robot_s_gray = convertToGray(robot_s)
    plot(robot_s, robot_s_gray)
    pss_gray = convertToGray(pss)
    # plot(pss, pss_gray)

    
if __name__ == "__main__":
    main()