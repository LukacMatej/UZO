import cv2
import numpy as np
import argparse

def loadImage(path: str):
    return cv2.imread(path)

def closestNeighbor(image, x, y):
    h, w = image.shape[:2]
    x = round(x)
    y = round(y)
    return image[y, x]

def rotateImage(image, degree):
    (h, w) = image.shape[:2]
    
    center = (w // 2, h // 2)
    
    angle_rad = np.deg2rad(degree)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    new_w = int(abs(h * sin_angle) + abs(w * cos_angle))
    new_h = int(abs(h * cos_angle) + abs(w * sin_angle))
    
    new_center = (new_w // 2, new_h // 2)
    
    rotated = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    
    for y in range(new_h):
        for x in range(new_w):
            x_translated = x - new_center[0]
            y_translated = y - new_center[1]
            
            x_rotated = int(x_translated * cos_angle + y_translated * sin_angle + center[0])
            y_rotated = int(-x_translated * sin_angle + y_translated * cos_angle + center[1])
            
            if 0 <= x_rotated < w and 0 <= y_rotated < h:
                rotated[y, x] = closestNeighbor(image, x_rotated, y_rotated)
    return rotated
    

def main(args):
    image = loadImage("cv03_robot.bmp")
    if args.degree:
        degree = float(args.degree)
        rotated = rotateImage(image, degree)
    cv2.imshow("robot",rotated)
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parser_init() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-d", "--degree", type=float, help="Degree of rotation"
    )
    return argparser

if __name__=="__main__":
    argparser: argparse.ArgumentParser = parser_init()
    args: argparse.Namespace = argparser.parse_args()
    main(args)