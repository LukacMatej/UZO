import cv2 as opencv
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dctn, idctn 

def loadImages(images) -> list:
    imgs: list = []
    for file in os.listdir(images):
        if file.endswith(".jpg"):
            imgs.append(opencv.imread(os.path.join(images, file)))
    return imgs

def convertToGray(imgs: list) -> list:
    gray_imgs: list = []
    for img in imgs:
        gray_imgs.append(opencv.cvtColor(img, opencv.COLOR_BGR2GRAY))
    return gray_imgs

def convertToDCP(imgs_grey: list) -> list:
    dcp_imgs: list = []
    for img in imgs_grey:
        dcp_imgs.append(dctn(img))
    return dcp_imgs

def limitDCT(dcts, limit):
    limited_dcts = np.zeros_like(dcts)
    limited_dcts[:limit, :limit] = dcts[:limit, :limit]
    return limited_dcts

def main():
    images = "./cv04/images/"
    imgs = loadImages(images)
    imgs_grey = convertToGray(imgs)
    dcp_imgs = convertToDCP(imgs_grey)
    limited_dcts = [limitDCT(dcp, 5) for dcp in dcp_imgs]
    limited_imgs = [idctn(dct) for dct in limited_dcts]
    image_differences = []
    for img in limited_imgs:
        differences = []
        for img2 in limited_imgs:
            img = opencv.resize(img, (250, 250))
            img2 = opencv.resize(img2, (250, 250))
            difference = np.sum(np.abs(img - img2))
            differences.append(difference)
        image_differences.append(differences)

    sorted_images = []
    for i, diffs in enumerate(image_differences):
        sorted_indices = np.argsort(diffs)
        sorted_images.append([imgs[idx] for idx in sorted_indices])

    fig, axes = plt.subplots(9, 9, figsize=(15, 15))
    for i in range(9):
        for j in range(9):
            if j < len(sorted_images) and i < len(sorted_images[j]):
                axes[i, j].imshow(opencv.cvtColor(sorted_images[i][j], opencv.COLOR_BGR2RGB))
                axes[i, j].axis('off')
    plt.show()

if __name__ == "__main__":
    main()