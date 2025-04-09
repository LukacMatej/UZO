import cv2
import numpy as np
import matplotlib.pyplot as plt

def convertToGray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image

def thresholdImage(channel,threshold):
    _, binary = cv2.threshold(channel.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def colorRegions(thresh, min_size=90):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    count = 0
    valid_centroids = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            count += 1
            valid_centroids.append(centroids[i])
    return count, valid_centroids

def drawCentroids(image, centroids):
    image_with_centroids = image.copy()
    for x, y in centroids:
        x, y = int(round(x)), int(round(y))
        cv2.drawMarker(image_with_centroids, (x, y), (0, 255, 0), 
                       markerType=cv2.MARKER_CROSS, markerSize=8, 
                       thickness=1, line_type=cv2.LINE_AA)
    return image_with_centroids

def topHat(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return tophat


def plotImgs(original, top_hat, histogram_original, histogram_top_hat, centroids, rice_count):    
    plt.figure(figsize=(18, 12))  # Adjusted figsize for better spacing
    
    plt.subplot(231)
    plt.title('Originální obrázek')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(232)
    plt.title('Top hat')
    plt.imshow(top_hat, cmap='gray')
    plt.axis('off')
    
    plt.subplot(233)
    plt.title('Histogram originálního obrázku')
    plt.bar(range(256), histogram_original.ravel(), color='blue', width=0.5)
    plt.xlim([0, 256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('#')
    
    plt.subplot(234)
    plt.title('Histogram Top hat')
    plt.bar(range(256), histogram_top_hat.ravel(), color='blue', width=0.5)
    plt.xlim([0, 256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('#')
    
    plt.subplot(235)
    plt.imshow(cv2.cvtColor(centroids, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Number of rice grains: ' + str(rice_count))
    
    plt.tight_layout()
    plt.show()
    

def main():
    img_rice = loadImage("cv09_rice.bmp")
    rice_grey = convertToGray(img_rice)
    tophat_rice = topHat(rice_grey)
    histogram_original = cv2.calcHist([rice_grey], [0], None, [256], [0, 256])
    histogram_top_hat = cv2.calcHist([tophat_rice], [0], None, [256], [0, 256])
    rice_thresh = thresholdImage(rice_grey, 80)
    tophat_thresh = thresholdImage(tophat_rice, 0)
    count, info_rice = colorRegions(tophat_thresh)
    img_rice_centroids = drawCentroids(img_rice, info_rice)
    plotImgs(rice_thresh, tophat_thresh, histogram_original, histogram_top_hat, img_rice_centroids, count)
    
    
if __name__ == "__main__":
    main()