import cv2 as opencv
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

def load_image(images) -> list:
    imgs: list = []
    for file in os.listdir(images):
        if file.endswith(".jpg"):
            imgs.append(opencv.imread(os.path.join(images, file)))
    return imgs

def convert_rgb_to_gray(imgs: list) -> list:
    gray_imgs: list = []
    for img in imgs:
        gray_imgs.append(opencv.cvtColor(img, opencv.COLOR_BGR2GRAY))
    return gray_imgs

def get_histograms(imgs: list) -> list:
    histograms: list = []
    for img in imgs:
        histograms.append(opencv.calcHist([img], [0], None, [256], [0, 256]))
    return histograms

def main(images):
    imgs = load_image(images)
    imgs_grey = convert_rgb_to_gray(imgs)
    histograms = get_histograms(imgs_grey)
    differences = []
    image_differences = []
    for img in histograms:
        differences = []
        for img2 in histograms:
            difference = opencv.compareHist(img, img2, opencv.HISTCMP_CHISQR)
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

def parser_init() -> argparse.ArgumentParser:
    """
    Initialize the argument parser for the server.

    Returns:
        argparse.ArgumentParser: The argument parser object.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i", "--images", action="store"
    )
    return argparser

if __name__ == "__main__":
    parser: argparse.ArgumentParser = parser_init()
    args: argparse.Namespace = parser.parse_args()
    main(args.images)