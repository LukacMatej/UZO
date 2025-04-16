import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def convertToGray(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray

def loadImage(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")
    return image    

def loadImagesFromFolder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                if filename == "unknown.bmp":
                    continue
                image = loadImage(file_path)
                images.append(image)
            except FileNotFoundError:
                print(f"Could not load image: {file_path}")
    return images

def imageToVector(image):
    resized_image = cv2.resize(image, (64, 64))
    vector = resized_image.flatten()
    return vector

def createMatrixWp(vectors):
    Wp = np.column_stack(vectors)
    return Wp

def calculateAverageVector(Wp):
    wp = np.mean(Wp, axis=1)
    return wp

def createMatrixW(Wp, wp):
    W = Wp - wp[:, np.newaxis]
    return W

def createCovarianceMatrix(W):
    C = np.dot(W.T, W)
    return C

def calculateEigenvaluesAndEigenvectors(C):
    eigenvalues, eigenvectors = np.linalg.eig(C)
    return eigenvalues, eigenvectors

def createEpMatrix(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    Ep = sorted_eigenvectors[:, :9]
    return Ep

def createEigenSpace(W, Ep):
    E = np.dot(W, Ep)
    return E

def projectVectorsToEigenSpace(E, W):
    PI = np.dot(E.T, W)
    return PI

def processUnknownImage(unknown_image, avg_vector, E, PI):
    # Step 1: Convert unknown image to grayscale and create vector wpu
    unknown_gray = convertToGray(unknown_image)
    wpu = imageToVector(unknown_gray)
    
    # Step 2: Calculate wu = wpu - wp
    wu = wpu - avg_vector
    
    # Step 3: Project unknown vector PT = ET * wu
    PT = np.dot(E.T, wu)
    
    # Step 4: Compare known feature vectors PI(i) and unknown PT using minimal distance
    distances = np.linalg.norm(PI - PT[:, np.newaxis], axis=0)
    min_distance_index = np.argmin(distances)
    return min_distance_index, distances[min_distance_index]

def plotImages(unknown_image, images, matched_index):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB))
    plt.title("Unknown Image")
    plt.axis("off")
    matched_image = images[matched_index]
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.title("Matched Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    images = loadImagesFromFolder("./images/")
    images_gray = []
    for image in images:
        images_gray.append(convertToGray(image))
    images_vectors = [imageToVector(image) for image in images_gray]
    images_matrix = createMatrixWp(images_vectors)
    avg_vector = calculateAverageVector(images_matrix)
    W = createMatrixW(images_matrix,avg_vector)
    cov_matrix = createCovarianceMatrix(W)
    eigen_values = calculateEigenvaluesAndEigenvectors(cov_matrix)
    Ep = createEpMatrix(eigen_values[0],eigen_values[1])
    E = createEigenSpace(W, Ep)
    Pi = projectVectorsToEigenSpace(E,W)
    
    #testovaci
    unknown = loadImage("./images/unknown.bmp")
    unknown_gray = convertToGray(unknown)
    unknown_vector = imageToVector(unknown_gray)
    results = processUnknownImage(unknown_gray,avg_vector,E,Pi)
    plotImages(unknown,images,results[0])
    
if __name__ == "__main__":
    main()