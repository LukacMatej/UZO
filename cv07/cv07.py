import cv2
import numpy as np
import matplotlib.pyplot as plt

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image

def calculateGreenChannel(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = np.float32(img_rgb[:, :, 0]), np.float32(img_rgb[:, :, 1]), np.float32(img_rgb[:, :, 2])
    return 255 - ((G * 255) / (R + G + B))

import numpy as np

def threshold_image(green_channel):
    green_channel = green_channel.astype(np.uint8)
    
    hist, _ = np.histogram(green_channel, bins=256, range=[0, 256])
    
    total_pixels = green_channel.size
    
    max_variance = 0
    optimal_threshold = 0
    
    cumulative_sum = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))
    
    for threshold in range(1, 256):
        bg_pixels = cumulative_sum[threshold-1]
        bg_prob = bg_pixels / total_pixels
        
        fg_pixels = total_pixels - bg_pixels
        fg_prob = 1 - bg_prob
        
        if bg_pixels == 0 or fg_pixels == 0:
            continue
        
        bg_mean = cumulative_mean[threshold-1] / bg_pixels if bg_pixels > 0 else 0
        fg_mean = (cumulative_mean[-1] - cumulative_mean[threshold-1]) / fg_pixels if fg_pixels > 0 else 0

        variance = bg_prob * fg_prob * (bg_mean - fg_mean) ** 2
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = threshold
    binary = np.zeros_like(green_channel)
    binary[green_channel >= optimal_threshold] = 255
    return binary

def color_regions(binary):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    labels_filtered = labels.copy()
    coins_info = []
    for i in range(1, num_labels):
        coin_type = "5" if stats[i, cv2.CC_STAT_AREA] > 4000 else "1"
        coins_info.append({
            'label': i,
            'type': coin_type,
            'area': stats[i, cv2.CC_STAT_AREA],
            'centroid': centroids[i]
        })
    return labels_filtered, coins_info

def draw_centroids(image, coins_info):
    image_with_centroids = image.copy()
    total_value = 0
    for coin in coins_info:
        x, y = map(int, coin['centroid'])
        cv2.circle(image_with_centroids, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(image_with_centroids, 
                    f"{coin['type']}", 
                    (x+10, y+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 255), 
                    2)
        total_value += int(coin['type'])
    return image_with_centroids, total_value

def plotResults(img_orig, green_channel, labels, img_with_centroids, total_value):
    plt.figure(figsize=(16, 4))
    
    plt.subplot(141)
    plt.title('Originální obrázek')
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(142)
    plt.title('Zelená složka')
    plt.imshow(green_channel, cmap='gray')
    plt.axis('off')

    plt.subplot(143)
    plt.title('Označené oblasti')
    plt.imshow(labels, cmap='nipy_spectral')
    plt.axis('off')

    plt.subplot(144)
    plt.title('Detekované mince')
    plt.imshow(cv2.cvtColor(img_with_centroids, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Celková hodnota mincí: {total_value} CZK")

def main():
    image_path = "./cv07/cv07_segmentace.bmp"
    image = loadImage(image_path)
    green_channel = calculateGreenChannel(image)
    binary = threshold_image(green_channel)
    labels, coins_info = color_regions(binary)
    img_with_centroids, total_value = draw_centroids(image, coins_info)
    plotResults(image, green_channel, labels, img_with_centroids, total_value)

if __name__ == "__main__":
    main()