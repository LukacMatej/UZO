import cv2
import numpy as np
import matplotlib.pyplot as plt


def convertToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image

    