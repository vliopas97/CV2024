# TUWIEN - WS2024 CV: Task3 - Scene recognition using Bag of Visual Words
# *********+++++++++*******++++GROUP NO. 06
from typing import List
import sklearn
import sklearn.metrics.pairwise as sklearn_pairwise
import cv2
import numpy as np
import random
import time

import sklearn.preprocessing


def extract_dsift(images: List[np.ndarray], stepsize: int, num_samples: int = None) -> List[np.ndarray]:
    """
    Extracts dense feature points on a regular grid with 'stepsize' and optionally returns
    'num_samples' random samples per image. If 'num_samples' is not provided, it takes all
    features extracted with the given 'stepsize'. SIFT.compute has the argument "keypoints",
    which should be set to a list of keypoints for each square.
    
    Args:
    - images (List[np.ndarray]): List of images to extract dense SIFT features [num_of_images x n x m] - float
    - stepsize (int): Grid spacing, step size in x and y direction.
    - num_samples (int, optional): Random number of samples per image.

    Returns:
    - List[np.ndarray]: SIFT descriptors for each image [number_of_images x num_samples x 128] - float
    """

    tic = time.perf_counter()

    # student_code start
    
    sift = cv2.SIFT_create()
    
    all_descriptors = []
    
    for img in images:
        # Create keypoints on a regular grid
        keypoints = []
        h, w = img.shape
        for y in range(0, h, stepsize):
            for x in range(0, w, stepsize):
                keypoints.append(cv2.KeyPoint(x, y, stepsize))

        if num_samples is not None:
            keypoints = random.sample(keypoints, min(num_samples, len(keypoints)))
        
        img = img.astype(np.uint8)
        _, descriptors = sift.compute(img, keypoints)

        all_descriptors.append(descriptors)
            
    # student_code end

    toc = time.perf_counter()
    print("DSIFT Extraction:", toc - tic, " seconds")
    # all_descriptors : list sift descriptors per image [number_of_images x num_samples x 128] - float
    return all_descriptors


def count_visual_words(dense_feat: List[np.ndarray], centroids: List[np.ndarray]) -> List[np.ndarray]:
    """
    For classification, generates a histogram of word occurrences per image.
    Utilizes sklearn_pairwise.pairwise_distances(..) to assign the descriptors per image
    to the nearest centroids and counts the occurrences of each centroid. The histogram
    should be as long as the vocabulary size (number of centroids).

    Args:
    - dense_feat (List[np.ndarray]): List of SIFT descriptors per image [number_of_images x num_samples x 128] - float
    - centroids (List[np.ndarray]): Centroids of clusters [vocabulary_size x 128]

    Returns:
    - List[np.ndarray]: List of histograms per image [number_of_images x vocabulary_size]
    """
    tic = time.perf_counter()

    # student_code start
    
    histograms = []  # To store histograms for each image

    for descriptors in dense_feat:
        histogram = np.zeros(centroids.shape[0], dtype=int)

        distances = sklearn_pairwise.pairwise_distances(descriptors, centroids)        
        closest_centroids = np.argmin(distances, axis=1)
        
        for index in closest_centroids:
            histogram[index] += 1
        
        histograms.append(histogram)


    # student_code end

    toc = time.perf_counter()
    print("Counting visual words:", toc - tic, " seconds")

    # histograms : list of histograms per image [number_of_images x vocabulary_size]
    return histograms


def calculate_vlad_descriptors(dense_feat: List[np.ndarray], centroids: List[np.ndarray]) -> List[np.ndarray]:
    """
    For classification, generate a histogram of word occurence per image
     Use sklearn_pairwise.pairwise_distances(..) to assign the descriptors per image
     to the nearest centroids and calculate for each word the residual to the nearest centroid
     The final feature vector should be as long as the vocabulary size (number of centroids) x feature dimension
     L2-normalize the final descriptors via sklearn.preprocessing.normalize.
     
    Args:
    - dense_feat : list sift descriptors per image [number_of_images x num_samples x 128] - float
    - centroids : centroids of clusters [vocabulary_size x 128]

    Returns:
    - List[np.ndarray]: List of histograms per image [number_of_images x (vocabulary_size x feature dimension)]
    """
    tic = time.perf_counter()

    
    # student_code start
    
    image_descriptors = []
    num_clusters, descriptor_dim = centroids.shape

    for img_descriptors in dense_feat:
        vlad_vector = np.zeros((num_clusters, descriptor_dim), dtype=np.float32)

        distances = sklearn_pairwise.pairwise_distances(img_descriptors, centroids)
        closest_centroids = np.argmin(distances, axis=1)

        for i, cluster_idx in enumerate(closest_centroids):
            residual = img_descriptors[i] - centroids[cluster_idx]
            vlad_vector[cluster_idx] += residual

        vlad_vector = vlad_vector.flatten()
        vlad_vector = sklearn.preprocessing.normalize(vlad_vector.reshape(1, -1), norm='l2').flatten()

        image_descriptors.append(vlad_vector)

    # student_code end
    

    toc = time.perf_counter()
    print("Counting visual words:", toc - tic, " seconds")

    # histograms : list of histograms per image [number_of_images x vocabulary_size]
    return image_descriptors